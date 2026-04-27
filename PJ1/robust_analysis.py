import gzip
import os
from struct import unpack

import numpy as np


SEED = 309
IMAGE_SIZE = 28
SOURCE_DIR = os.path.join("dataset", "MNIST")
OUTPUT_DIR = os.path.join("dataset", "MNIST_robust")


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


def transform_images(images, transform, seed=SEED):
    rng = np.random.default_rng(seed)
    output = np.zeros_like(images, dtype=np.float32)

    for i in range(images.shape[0]):
        if transform == "rotate":
            output[i] = rotate_image(images[i], rng.uniform(-15.0, 15.0))
        elif transform == "translate":
            output[i] = translate_image(images[i], rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0))
        else:
            output[i] = resize_image(images[i], rng.uniform(0.8, 1.2))

    return output.reshape(images.shape[0], -1).astype(np.float32)


def random_transform_images(images, seed=SEED):
    rng = np.random.default_rng(seed)
    output = np.zeros_like(images, dtype=np.float32)

    for i in range(images.shape[0]):
        transform = rng.choice(["rotate", "translate", "resize"])
        if transform == "rotate":
            output[i] = rotate_image(images[i], rng.uniform(-15.0, 15.0))
        elif transform == "translate":
            output[i] = translate_image(images[i], rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0))
        else:
            output[i] = resize_image(images[i], rng.uniform(0.8, 1.2))

    return output.reshape(images.shape[0], -1).astype(np.float32)


def add_gaussian_noise(images, sigma, seed=SEED):
    rng = np.random.default_rng(seed)
    noisy = images + rng.normal(0.0, sigma, size=images.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.reshape(images.shape[0], -1).astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = read_mnist_images(os.path.join(SOURCE_DIR, "t10k-images-idx3-ubyte.gz"))
    labels = read_mnist_labels(os.path.join(SOURCE_DIR, "t10k-labels-idx1-ubyte.gz"))

    np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), labels)

    for transform in ["rotate", "translate", "resize"]:
        transformed = transform_images(images, transform, seed=SEED)
        np.save(os.path.join(OUTPUT_DIR, f"test_{transform}.npy"), transformed)

    transformed = random_transform_images(images, seed=SEED)
    np.save(os.path.join(OUTPUT_DIR, "test_transform.npy"), transformed)

    for sigma in [0.05, 0.10, 0.20]:
        noisy = add_gaussian_noise(images, sigma, seed=SEED)
        np.save(os.path.join(OUTPUT_DIR, f"test_gaussian_{sigma:.2f}.npy"), noisy)

    print(f"saved robustness test sets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
