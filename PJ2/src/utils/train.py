from __future__ import annotations

from contextlib import contextmanager
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .losses import build_criterion
from .metrics import append_rows_to_csv, count_parameters
from .optim import build_optimizer, build_scheduler


def _effective_amp(cfg: dict, device: torch.device) -> bool:
    return bool(cfg.get("use_amp", True)) and device.type == "cuda"


def _make_grad_scaler(use_amp: bool):
    return torch.amp.GradScaler("cuda", enabled=use_amp)


def _autocast(device: torch.device, use_amp: bool):
    return torch.amp.autocast(device_type=device.type, enabled=use_amp)


def _clone_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _confusion_matrix(targets: list[int], predictions: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, pred in zip(targets, predictions):
        matrix[target][pred] += 1
    return matrix


def _augmentation_enabled(cfg: dict | bool | None) -> bool:
    if isinstance(cfg, dict):
        return bool(cfg.get("enabled", False))
    return bool(cfg)


def _augmentation_alpha(cfg: dict | bool | None, default: float) -> float:
    if isinstance(cfg, dict):
        return float(cfg.get("alpha", default))
    return default


def _rand_bbox(width: int, height: int, lam: float, device: torch.device) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = torch.randint(width, (1,), device=device).item()
    cy = torch.randint(height, (1,), device=device).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def _prepare_augmented_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    mixup_cfg: dict | bool | None,
    cutmix_cfg: dict | bool | None,
    device: torch.device,
):
    use_mixup = _augmentation_enabled(mixup_cfg)
    use_cutmix = _augmentation_enabled(cutmix_cfg)
    mixup_alpha = _augmentation_alpha(mixup_cfg, 0.2)
    cutmix_alpha = _augmentation_alpha(cutmix_cfg, 1.0)
    if use_mixup and use_cutmix:
        raise ValueError("mixup and cutmix cannot both be enabled in the same experiment")
    if use_mixup and mixup_alpha <= 0.0:
        raise ValueError("mixup alpha must be positive when mixup is enabled")
    if use_cutmix and cutmix_alpha <= 0.0:
        raise ValueError("cutmix alpha must be positive when cutmix is enabled")

    if use_mixup:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        index = torch.randperm(images.size(0), device=device)
        mixed_images = lam * images + (1.0 - lam) * images[index]
        return mixed_images, labels, labels[index], lam

    if use_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        index = torch.randperm(images.size(0), device=device)
        mixed_images = images.clone()
        _, _, height, width = mixed_images.shape
        x1, y1, x2, y2 = _rand_bbox(width, height, lam, device)
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
        return mixed_images, labels, labels[index], lam

    return images, labels, None, 1.0


def _mixed_loss(criterion, logits, labels_a, labels_b, lam: float):
    if labels_b is None:
        return criterion(logits, labels_a)
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def _mixed_correct(logits, labels_a, labels_b, lam: float) -> float:
    preds = logits.argmax(dim=1)
    if labels_b is None:
        return float((preds == labels_a).sum().item())
    return (
        lam * (preds == labels_a).sum().item()
        + (1.0 - lam) * (preds == labels_b).sum().item()
    )


def _grad_norm(parameters, device: torch.device) -> torch.Tensor:
    norms = [
        p.grad.detach().norm(2)
        for p in parameters
        if p.grad is not None
    ]
    if not norms:
        return torch.zeros((), device=device)
    return torch.norm(torch.stack(norms), 2)


def _clone_gradients(model: nn.Module) -> list[torch.Tensor | None]:
    return [
        None if p.grad is None else p.grad.detach().clone()
        for p in model.parameters()
    ]


def _grad_list_norm(grads: list[torch.Tensor | None], device: torch.device) -> torch.Tensor:
    norms = [grad.norm(2) for grad in grads if grad is not None]
    if not norms:
        return torch.zeros((), device=device)
    return torch.norm(torch.stack(norms), 2)


def _grad_difference_norm(
    grads_a: list[torch.Tensor | None],
    grads_b: list[torch.Tensor | None],
    device: torch.device,
) -> float:
    diffs = []
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None and grad_b is None:
            continue
        if grad_a is None:
            diffs.append(grad_b.norm(2))
        elif grad_b is None:
            diffs.append(grad_a.norm(2))
        else:
            diffs.append((grad_b - grad_a).norm(2))
    if not diffs:
        return 0.0
    return torch.norm(torch.stack(diffs), 2).item()


def _apply_grad_perturbation(
    model: nn.Module,
    grads: list[torch.Tensor | None],
    rho: float,
    grad_norm: torch.Tensor,
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    eps: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    scale = rho / (grad_norm + 1e-12)
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            if grad is None:
                continue
            epsilon = grad * scale
            param.add_(epsilon)
            eps.append((param, epsilon))
    return eps


def _restore_perturbation(eps: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    with torch.no_grad():
        for param, epsilon in eps:
            param.sub_(epsilon)


@contextmanager
def _disable_bn_running_stats(model: nn.Module):
    states = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            states.append((module, module.track_running_stats))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in states:
            module.track_running_stats = track_running_stats


def snapshot_bn_buffers(model: nn.Module) -> dict[nn.Module, dict[str, torch.Tensor]]:
    snapshot = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            buffers = {}
            for name in ["running_mean", "running_var", "num_batches_tracked"]:
                value = getattr(module, name, None)
                if value is not None:
                    buffers[name] = value.detach().clone()
            if buffers:
                snapshot[module] = buffers
    return snapshot


def restore_bn_buffers(snapshot: dict[nn.Module, dict[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for module, buffers in snapshot.items():
            for name, value in buffers.items():
                getattr(module, name).copy_(value)


def _save_checkpoint(
    path: str | Path,
    state_dict: dict,
    experiment_id: str,
    cfg: dict,
    epoch: int,
    val_acc: float,
    params: int,
    task: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "experiment_id": experiment_id,
            "model_cfg": cfg.get("model", {}),
            "train_cfg": cfg.get("train", {}),
            "epoch": epoch,
            "val_acc": val_acc,
            "params": params,
            "task": task,
        },
        path,
    )


def train_one_epoch_standard(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    mixup_cfg: dict | bool | None = None,
    cutmix_cfg: dict | bool | None = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_inputs, labels_a, labels_b, lam = _prepare_augmented_batch(
            images,
            labels,
            mixup_cfg,
            cutmix_cfg,
            device,
        )
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, use_amp):
            logits = model(batch_inputs)
            loss = _mixed_loss(criterion, logits, labels_a, labels_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += _mixed_correct(logits, labels_a, labels_b, lam)
        total_items += batch_size

    return total_loss / total_items, 100.0 * total_correct / total_items


def train_one_epoch_sam(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    rho: float,
    mixup_cfg: dict | bool | None = None,
    cutmix_cfg: dict | bool | None = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    first_scaler = _make_grad_scaler(use_amp)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_inputs, labels_a, labels_b, lam = _prepare_augmented_batch(
            images,
            labels,
            mixup_cfg,
            cutmix_cfg,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(batch_inputs)
            loss = _mixed_loss(criterion, logits, labels_a, labels_b, lam)
        first_scaler.scale(loss).backward()
        first_scaler.unscale_(optimizer)
        grad_norm = _grad_norm(model.parameters(), device)
        first_grads = _clone_gradients(model)
        eps = _apply_grad_perturbation(model, first_grads, rho, grad_norm)
        optimizer.zero_grad(set_to_none=True)
        first_scaler.update()

        with _disable_bn_running_stats(model):
            with _autocast(device, use_amp):
                logits_adv = model(batch_inputs)
                loss_adv = _mixed_loss(criterion, logits_adv, labels_a, labels_b, lam)
        scaler.scale(loss_adv).backward()
        _restore_perturbation(eps)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += _mixed_correct(logits, labels_a, labels_b, lam)
        total_items += batch_size

    return total_loss / total_items, 100.0 * total_correct / total_items


def _gradient_snapshot_for_batch(
    model,
    criterion,
    images,
    labels,
    device,
    use_amp: bool,
) -> tuple[float, list[torch.Tensor | None], torch.Tensor]:
    model.zero_grad(set_to_none=True)
    with _autocast(device, use_amp):
        logits = model(images)
        loss = criterion(logits, labels)
    loss.backward()
    grads = _clone_gradients(model)
    return loss.item(), grads, _grad_list_norm(grads, device)


def compute_bn_analysis_metrics(
    model,
    criterion,
    images,
    labels,
    device,
    use_amp: bool,
    distances: list[float],
    sharpness_rho: float,
) -> dict:
    snapshot = snapshot_bn_buffers(model)
    grad_predict_values: list[float] = []
    beta_values: list[float] = []
    sam_sharpness = ""

    try:
        restore_bn_buffers(snapshot)
        base_loss, base_grads, base_grad_norm = _gradient_snapshot_for_batch(
            model,
            criterion,
            images,
            labels,
            device,
            use_amp,
        )

        for distance in distances:
            eps = _apply_grad_perturbation(model, base_grads, distance, base_grad_norm)
            restore_bn_buffers(snapshot)
            perturbed_loss, perturbed_grads, _ = _gradient_snapshot_for_batch(
                model,
                criterion,
                images,
                labels,
                device,
                use_amp,
            )
            _restore_perturbation(eps)
            restore_bn_buffers(snapshot)

            grad_diff = _grad_difference_norm(base_grads, perturbed_grads, device)
            grad_predict_values.append(grad_diff)
            beta_values.append(grad_diff / distance)
            if abs(distance - sharpness_rho) < 1e-12:
                sam_sharpness = perturbed_loss - base_loss

        if sam_sharpness == "":
            eps = _apply_grad_perturbation(model, base_grads, sharpness_rho, base_grad_norm)
            restore_bn_buffers(snapshot)
            perturbed_loss, _, _ = _gradient_snapshot_for_batch(
                model,
                criterion,
                images,
                labels,
                device,
                use_amp,
            )
            _restore_perturbation(eps)
            sam_sharpness = perturbed_loss - base_loss
    finally:
        restore_bn_buffers(snapshot)
        model.zero_grad(set_to_none=True)

    return {
        "grad_predict_min": min(grad_predict_values),
        "grad_predict_max": max(grad_predict_values),
        "beta_smooth_min": min(beta_values),
        "beta_smooth_max": max(beta_values),
        "sam_sharpness": sam_sharpness,
    }


def train_one_epoch_with_steps(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    epoch: int,
    start_global_step: int,
    start_time: float,
    analysis_cfg: dict | None = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    global_step = start_global_step
    step_rows: list[dict] = []
    analysis_cfg = analysis_cfg or {}
    analysis_enabled = bool(analysis_cfg.get("enabled", False))
    analysis_every = int(analysis_cfg.get("every_n_steps", 24))
    analysis_distances = [float(x) for x in analysis_cfg.get("distances", [0.01, 0.02, 0.05])]
    analysis_rho = float(analysis_cfg.get("rho", 0.05))

    for step, (images, labels) in enumerate(loader, start=1):
        step_start = time.time()
        global_step += 1
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        preds = logits.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_items += batch_size
        row = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "train_loss": loss.item(),
            "train_acc": 100.0 * batch_correct / batch_size,
            "step_time_sec": time.time() - step_start,
            "total_time_sec": time.time() - start_time,
            "grad_predict_min": "",
            "grad_predict_max": "",
            "beta_smooth_min": "",
            "beta_smooth_max": "",
            "sam_sharpness": "",
        }
        if analysis_enabled and global_step % analysis_every == 0:
            row.update(
                compute_bn_analysis_metrics(
                    model,
                    criterion,
                    images,
                    labels,
                    device,
                    use_amp,
                    analysis_distances,
                    analysis_rho,
                )
            )
        step_rows.append(row)

    return (
        total_loss / total_items,
        100.0 * total_correct / total_items,
        step_rows,
        global_step,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_items += batch_size
        predictions.extend(preds.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return (
        total_loss / total_items,
        100.0 * total_correct / total_items,
        predictions,
        targets,
    )


def _supervised_row(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    best_epoch: int,
    epoch_time: float,
    total_time: float,
) -> dict:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epoch_time_sec": epoch_time,
        "total_time_sec": total_time,
    }


def fit_supervised_experiment(
    experiment_id: str,
    model,
    train_loader,
    val_loader,
    test_loader,
    cfg: dict,
    train_csv: str | Path,
    test_json: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    task: str,
    class_names: list[str],
) -> dict:
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    criterion = build_criterion(
        train_cfg.get("loss", "cross_entropy"),
        float(train_cfg.get("label_smoothing", 0.0)),
    )
    model.to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, epochs)
    use_amp = _effective_amp(train_cfg, device)
    scaler = _make_grad_scaler(use_amp)
    params = count_parameters(model)

    rows: list[dict] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_state = _clone_state_dict(model)
    start_time = time.time()

    progress = tqdm(range(1, epochs + 1), desc=experiment_id, unit="epoch")
    for epoch in progress:
        epoch_start = time.time()
        sam_cfg = train_cfg.get("sam", {})
        if isinstance(sam_cfg, dict) and bool(sam_cfg.get("enabled", False)):
            sam_use_amp = use_amp and bool(sam_cfg.get("use_amp", False))
            train_loss, train_acc = train_one_epoch_sam(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                _make_grad_scaler(sam_use_amp),
                sam_use_amp,
                float(sam_cfg.get("rho", 0.05)),
                train_cfg.get("mixup"),
                train_cfg.get("cutmix"),
            )
        else:
            train_loss, train_acc = train_one_epoch_standard(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                use_amp,
                train_cfg.get("mixup"),
                train_cfg.get("cutmix"),
            )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = _clone_state_dict(model)
            _save_checkpoint(
                checkpoint_path,
                best_state,
                experiment_id,
                cfg,
                epoch,
                val_acc,
                params,
                task,
            )

        row = _supervised_row(
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            best_val_acc,
            best_epoch,
            time.time() - epoch_start,
            time.time() - start_time,
        )
        rows.append(row)
        progress.set_postfix(train_acc=f"{train_acc:.2f}", val_acc=f"{val_acc:.2f}")

    model.load_state_dict(best_state)
    test_loss, test_acc, predictions, targets = evaluate(
        model, test_loader, criterion, device, use_amp
    )
    append_rows_to_csv(train_csv, rows)

    num_classes = len(class_names)
    test_result = {
        "task": task,
        "experiment_id": experiment_id,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_error": 100.0 - test_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "params": params,
        "use_amp": use_amp,
        "model": cfg.get("model", {}),
        "train": cfg.get("train", {}),
        "classes": class_names,
        "confusion_matrix": _confusion_matrix(targets, predictions, num_classes),
    }
    _write_json(test_json, test_result)

    return {
        "experiment_id": experiment_id,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_error": 100.0 - test_acc,
        "params": params,
        "confusion_matrix": test_result["confusion_matrix"],
    }


def fit_step_experiment(
    experiment_id: str,
    model,
    train_loader,
    test_loader,
    cfg: dict,
    train_csv: str | Path,
    test_json: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    task: str,
    class_names: list[str],
) -> dict:
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    criterion = build_criterion(
        train_cfg.get("loss", "cross_entropy"),
        float(train_cfg.get("label_smoothing", 0.0)),
    )
    model.to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, epochs)
    use_amp = _effective_amp(train_cfg, device)
    scaler = _make_grad_scaler(use_amp)
    params = count_parameters(model)

    start_time = time.time()
    global_step = 0
    final_train_loss = 0.0
    final_train_acc = 0.0

    progress = tqdm(range(1, epochs + 1), desc=experiment_id, unit="epoch")
    for epoch in progress:
        epoch_start = time.time()
        train_loss, train_acc, step_rows, global_step = train_one_epoch_with_steps(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            epoch,
            global_step,
            start_time,
            cfg.get("analysis", {}),
        )
        if scheduler is not None:
            scheduler.step()

        final_train_loss = train_loss
        final_train_acc = train_acc
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        for row in step_rows:
            row.update(
                {
                    "epoch_train_loss": "",
                    "epoch_train_acc": "",
                    "epoch_time_sec": "",
                }
            )
        if step_rows:
            step_rows[-1].update(
                {
                    "epoch_train_loss": train_loss,
                    "epoch_train_acc": train_acc,
                    "epoch_time_sec": epoch_time,
                    "total_time_sec": total_time,
                }
            )
        append_rows_to_csv(train_csv, step_rows)
        progress.set_postfix(train_acc=f"{train_acc:.2f}")

    if bool(cfg.get("output", {}).get("save_checkpoint", False)):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "experiment_id": experiment_id,
                "model_cfg": cfg.get("model", {}),
                "train_cfg": cfg.get("train", {}),
                "epoch": epochs,
                "params": params,
                "task": task,
                "checkpoint_type": "final",
            },
            checkpoint_path,
        )

    test_loss, test_acc, predictions, targets = evaluate(
        model, test_loader, criterion, device, use_amp
    )
    num_classes = len(class_names)
    test_result = {
        "task": task,
        "experiment_id": experiment_id,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_error": 100.0 - test_acc,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "params": params,
        "use_amp": use_amp,
        "model": cfg.get("model", {}),
        "train": cfg.get("train", {}),
        "classes": class_names,
        "confusion_matrix": _confusion_matrix(targets, predictions, num_classes),
    }
    _write_json(test_json, test_result)

    return {
        "experiment_id": experiment_id,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_error": 100.0 - test_acc,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "params": params,
        "confusion_matrix": test_result["confusion_matrix"],
    }
