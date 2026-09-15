import os
import cv2
import numpy as np

from preprocessing import divide_into_blocks
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


print("=" * 70)
print("AI-ADAPTIVE EPVD: 5–7 BIT PREDICTION TEST")
print("=" * 70)


# ============================================================
# Load Lena
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


print(
    f"\nImage shape: {image.shape}"
)


# ============================================================
# Divide into blocks
# ============================================================

blocks = divide_into_blocks(
    image,
    block_size=8
)

print(
    f"Total blocks: {len(blocks)}"
)


# ============================================================
# Load AI model
# ============================================================

predictor = AdaptiveBitPredictor()


# ============================================================
# Predict
# ============================================================

predictions = predictor.predict_blocks(
    blocks
)


# ============================================================
# Distribution
# ============================================================

unique, counts = np.unique(
    predictions,
    return_counts=True
)

print("\nAI predicted bit distribution:")

for bits, count in zip(
    unique,
    counts
):

    percentage = (
        count
        /
        len(predictions)
        *
        100
    )

    print(
        f"{bits} bits : "
        f"{count} blocks "
        f"({percentage:.2f}%)"
    )


# ============================================================
# Statistics
# ============================================================

print("\nPrediction statistics:")

print(
    f"Minimum bits : {predictions.min()}"
)

print(
    f"Maximum bits : {predictions.max()}"
)

print(
    f"Average bits : {predictions.mean():.4f}"
)


# ============================================================
# Sample predictions
# ============================================================

print("\nFirst 20 block predictions:")

print(
    predictions[:20]
)


# ============================================================
# Validation
# ============================================================

valid = np.all(
    np.isin(
        predictions,
        [5, 6, 7]
    )
)

print("\nAll predictions valid:", valid)

print("=" * 70)