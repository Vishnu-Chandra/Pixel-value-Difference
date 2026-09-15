import cv2
import numpy as np
from pathlib import Path
import sys


def load_image(image_path):
    """Load an RGB image."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_ycbcr(image):
    """Convert RGB image to YCbCr."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb(image):
    """Convert YCbCr image back to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)


def divide_into_blocks(channel, block_size=8):
    """Divide a channel into non-overlapping blocks."""
    height, width = channel.shape

    height = height - (height % block_size)
    width = width - (width % block_size)

    channel = channel[:height, :width]

    blocks = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = channel[
                i:i + block_size,
                j:j + block_size
            ]
            blocks.append(block.copy())

    return blocks


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
        print("Example: python src/preprocessing.py data/cover/test.png")
        raise SystemExit(0)

    image_path = Path(sys.argv[1])
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    image = load_image(str(image_path))

    ycbcr = rgb_to_ycbcr(image)

    Y, Cr, Cb = cv2.split(ycbcr)

    blocks = divide_into_blocks(Y, 8)

    print("Image shape:", image.shape)
    print("YCbCr shape:", ycbcr.shape)
    print("Y channel shape:", Y.shape)
    print("Number of 8x8 blocks:", len(blocks))
    print("First block shape:", blocks[0].shape)