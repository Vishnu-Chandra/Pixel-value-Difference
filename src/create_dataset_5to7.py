import os
import cv2
import numpy as np
import pandas as pd

from features import extract_features


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "labeled_features_5to7.csv"
)


# ============================================================
# Images
# ============================================================

IMAGE_NAMES = [
    "lena.png",
    "baboon.jpeg",
    "car-house.jpeg",
    "jet.jpeg",
    "peppers.jpeg",
    "sailboat.jpeg"
]


# ============================================================
# Load images
# ============================================================

all_rows = []

print("=" * 70)
print("Creating 5–7 Bit Adaptive Dataset")
print("=" * 70)


for image_name in IMAGE_NAMES:

    image_path = os.path.join(
        COVER_DIR,
        image_name
    )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load: {image_path}"
        )

    height, width = image.shape

    block_size = 8

    height = height - height % block_size
    width = width - width % block_size

    image = image[:height, :width]

    blocks = []

    for i in range(
        0,
        height,
        block_size
    ):
        for j in range(
            0,
            width,
            block_size
        ):

            block = image[
                i:i + block_size,
                j:j + block_size
            ]

            blocks.append(block.copy())

    print(
        f"{image_name}: {len(blocks)} blocks"
    )

    for block_id, block in enumerate(blocks):

        features = extract_features(block)

        all_rows.append(
            [
                image_name,
                block_id,
                *features
            ]
        )


# ============================================================
# Create dataframe
# ============================================================

feature_columns = [
    "image_name",
    "block_id",
    "variance",
    "entropy",
    "edge_density",
    "gradient_magnitude",
    "mean_pixel_difference",
    "max_pixel_difference",
    "std_pixel_difference"
]

df = pd.DataFrame(
    all_rows,
    columns=feature_columns
)


# ============================================================
# Normalize features
# ============================================================

numeric_features = [
    "variance",
    "entropy",
    "edge_density",
    "gradient_magnitude",
    "mean_pixel_difference",
    "max_pixel_difference",
    "std_pixel_difference"
]

normalized = pd.DataFrame(index=df.index)

for column in numeric_features:

    minimum = df[column].min()
    maximum = df[column].max()

    if maximum == minimum:
        normalized[column] = 0.0

    else:
        normalized[column] = (
            (df[column] - minimum)
            /
            (maximum - minimum)
        )


# ============================================================
# Texture score
# ============================================================

weights = {
    "variance": 0.20,
    "entropy": 0.15,
    "edge_density": 0.25,
    "gradient_magnitude": 0.20,
    "mean_pixel_difference": 0.10,
    "max_pixel_difference": 0.05,
    "std_pixel_difference": 0.05
}

df["texture_score"] = 0.0

for feature, weight in weights.items():

    df["texture_score"] += (
        normalized[feature] * weight
    )


# ============================================================
# Determine thresholds
# ============================================================

q25 = df["texture_score"].quantile(0.25)
q50 = df["texture_score"].quantile(0.50)

print("\nTexture thresholds:")
print(f"Q25: {q25}")
print(f"Q50: {q50}")


# ============================================================
# Assign adaptive embedding bits
#
# Smooth       → 5 bits
# Medium       → 6 bits
# High         → 7 bits
# ============================================================

def assign_bits(score):

    if score <= q25:
        return 5

    elif score <= q50:
        return 6

    else:
        return 7


df["embedding_bits"] = (
    df["texture_score"]
    .apply(assign_bits)
)


# ============================================================
# Distribution
# ============================================================

print("\nEmbedding-bit distribution:")

distribution = (
    df["embedding_bits"]
    .value_counts()
    .sort_index()
)

print(distribution)


print("\nPercentages:")

percentages = (
    df["embedding_bits"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print(percentages)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nDataset shape:", df.shape)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nColumns:")
print(df.columns.tolist())

print("=" * 70)