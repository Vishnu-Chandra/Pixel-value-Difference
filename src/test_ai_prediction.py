import os
import sys
import numpy as np

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from preprocessing import (
    load_image,
    rgb_to_ycbcr,
    divide_into_blocks
)

from ai_adaptive import AdaptiveBitPredictor


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover",
    "lena.png"
)


print("----------------------------------------")
print("Testing AI Adaptive Prediction")
print("----------------------------------------")
print()


predictor = AdaptiveBitPredictor()

print(
    "Random Forest loaded successfully."
)

print()


image = load_image(
    IMAGE_PATH
)

ycbcr = rgb_to_ycbcr(
    image
)

Y = ycbcr[:, :, 0]

blocks = divide_into_blocks(
    Y,
    block_size=8
)


predictions = predictor.predict_blocks(
    blocks
)


print(
    "Image:",
    os.path.basename(IMAGE_PATH)
)

print(
    "Total blocks:",
    len(blocks)
)

print()

print(
    "First 20 predictions:"
)

print(
    predictions[:20]
)

print()

print(
    "Prediction distribution:"
)

unique, counts = np.unique(
    predictions,
    return_counts=True
)

for bits, count in zip(
    unique,
    counts
):

    percentage = (
        count / len(predictions)
    ) * 100

    print(
        f"{bits} bits: "
        f"{count} blocks "
        f"({percentage:.2f}%)"
    )

print()

print(
    "Average predicted bits:",
    f"{np.mean(predictions):.3f}"
)

print()

print(
    "AI prediction test completed."
)