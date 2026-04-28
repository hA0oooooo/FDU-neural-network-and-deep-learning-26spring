import gzip
import os
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np


SEED = 309
IMAGE_SIZE = 28
SOURCE_DIR = os.path.join("dataset", "MNIST")
OUTPUT_DIR = os.path.join("dataset", "MNIST_augment")
FIGURE_DIR = os.path.join("outputs", "figures")


def read_mnist_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0


def read_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def bilinear_sample(image, x, y):
    h, w = image.shape

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    x0_clip = np.clip(x0, 0, w - 1)
    x1_clip = np.clip(x1, 0, w - 1)
    y0_clip = np.clip(y0, 0, h - 1)
    y1_clip = np.clip(y1, 0, h - 1)

    Ia = image[y0_clip, x0_clip]
    Ib = image[y1_clip, x0_clip]
    Ic = image[y0_clip, x1_clip]
    Id = image[y1_clip, x1_clip]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    sampled = wa * Ia + wb * Ib + wc * Ic + wd * Id
    sampled[~valid] = 0.0
    return sampled


def rotate_image(image, angle):
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    center = (IMAGE_SIZE - 1) / 2.0

    yy, xx = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="ij")
    x = xx - center
    y = yy - center

    src_x = cos_a * x + sin_a * y + center
    src_y = -sin_a * x + cos_a * y + center
    return bilinear_sample(image, src_x, src_y).astype(np.float32)


def translate_image(image, dx, dy):
    yy, xx = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="ij")
    src_x = xx - dx
    src_y = yy - dy
    return bilinear_sample(image, src_x, src_y).astype(np.float32)


def resize_image(image, scale):
    center = (IMAGE_SIZE - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="ij")
    src_x = (xx - center) / scale + center
    src_y = (yy - center) / scale + center
    return bilinear_sample(image, src_x, src_y).astype(np.float32)


def augment_image(image, rng):
    transform = rng.choice(["rotate", "translate", "resize"])
    if transform == "rotate":
        return rotate_image(image, rng.uniform(-15.0, 15.0))
    if transform == "translate":
        return translate_image(image, rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0))
    return resize_image(image, rng.uniform(0.8, 1.2))


def build_augmented_train_set(images, labels, prob=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    aug_images = images.copy()

    for i in range(images.shape[0]):
        if rng.random() < prob:
            aug_images[i] = augment_image(images[i], rng)

    return aug_images.reshape(images.shape[0], -1).astype(np.float32), labels.copy()


def save_examples(images):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = [
        ("rotation U(-15, 15)", "rotate"),
        ("translation U(-5, 5)", "translate"),
        ("resize U(0.8, 1.2)", "resize"),
    ]

    fig = plt.figure(figsize=(6, 5.0))
    gs = fig.add_gridspec(
        nrows=2 * len(rows),
        ncols=5,
        height_ratios=[1, 0.18] * len(rows),
        hspace=0.15,
        wspace=0.05
    )

    for row_id, (label, transform) in enumerate(rows):
        sample = images[rng.choice(images.shape[0], size=5, replace=False)]

        for col_id, image in enumerate(sample):
            ax = fig.add_subplot(gs[2 * row_id, col_id])

            if transform == "rotate":
                shown = rotate_image(image, rng.uniform(-15.0, 15.0))
            elif transform == "translate":
                shown = translate_image(image, rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0))
            else:
                shown = resize_image(image, rng.uniform(0.8, 1.2))

            ax.imshow(shown, cmap="gray")
            ax.axis("off")

        label_ax = fig.add_subplot(gs[2 * row_id + 1, :])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.04)
    fig.savefig(os.path.join(FIGURE_DIR, "dataaug.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = read_mnist_images(os.path.join(SOURCE_DIR, "train-images-idx3-ubyte.gz"))
    labels = read_mnist_labels(os.path.join(SOURCE_DIR, "train-labels-idx1-ubyte.gz"))

    aug_images, aug_labels = build_augmented_train_set(images, labels, prob=0.5)

    np.save(os.path.join(OUTPUT_DIR, "train_images.npy"), aug_images)
    np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), aug_labels)
    save_examples(images)

    print(f"saved augmented training set to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
