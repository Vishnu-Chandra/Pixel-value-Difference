import os
import cv2
import numpy as np
import pandas as pd

from preprocessing import divide_into_blocks
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
    "labeled_features_5to8.csv"
)

IMAGE_NAMES = [
    "lena.png",
    "baboon.jpeg",
    "car-house.jpeg",
    "jet.jpeg",
    "peppers.jpeg",
    "sailboat.jpeg"
]


FEATURE_COLUMNS = [
    "variance",
    "entropy",
    "edge_density",
    "gradient_magnitude",
    "mean_pixel_difference",
    "max_pixel_difference",
    "std_pixel_difference"
]


def calculate_texture_score(features):

    variance = features[0]
    entropy = features[1]
    edge_density = features[2]
    gradient = features[3]
    mean_diff = features[4]
    max_diff = features[5]
    std_diff = features[6]

    values = np.array([
        variance,
        entropy,
        edge_density,
        gradient,
        mean_diff,
        max_diff,
        std_diff
    ], dtype=np.float64)

    # Normalize using a local min-max style transformation.
    # The final dataset will be normalized globally below.
    return values


def main():

    rows = []

    print("=" * 70)
    print("Creating 5–8 Bit Adaptive Dataset")
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
            print(
                f"Could not load {image_name}"
            )
            continue

        blocks = divide_into_blocks(
            image,
            block_size=8
        )

        print(
            f"{image_name}: {len(blocks)} blocks"
        )

        for block_id, block in enumerate(blocks):

            features = extract_features(
                block
            )

            rows.append([
                image_name,
                block_id,
                *features
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "image_name",
            "block_id",
            *FEATURE_COLUMNS
        ]
    )

    print(
        "\nDataset shape:",
        df.shape
    )

    # ========================================================
    # GLOBAL FEATURE NORMALIZATION
    # ========================================================

    normalized = pd.DataFrame(
        index=df.index
    )

    for column in FEATURE_COLUMNS:

        values = df[column].astype(float)

        minimum = values.min()
        maximum = values.max()

        if maximum == minimum:
            normalized[column] = 0.0
        else:
            normalized[column] = (
                (values - minimum)
                /
                (maximum - minimum)
            )

    # ========================================================
    # WEIGHTED TEXTURE SCORE
    # ========================================================

    score = (
        0.20 * normalized["variance"]
        +
        0.15 * normalized["entropy"]
        +
        0.25 * normalized["edge_density"]
        +
        0.20 * normalized["gradient_magnitude"]
        +
        0.10 * normalized["mean_pixel_difference"]
        +
        0.05 * normalized["max_pixel_difference"]
        +
        0.05 * normalized["std_pixel_difference"]
    )

    df["texture_score"] = score

    # ========================================================
    # 5–8 BIT POLICY
    # ========================================================

    q25 = score.quantile(0.25)
    q50 = score.quantile(0.50)
    q75 = score.quantile(0.75)

    print("\nTexture thresholds:")
    print("Q25:", q25)
    print("Q50:", q50)
    print("Q75:", q75)

    def assign_bits(value):

        if value <= q25:
            return 5

        elif value <= q50:
            return 6

        elif value <= q75:
            return 7

        else:
            return 8

    df["embedding_bits"] = score.apply(
        assign_bits
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print("\nEmbedding-bit distribution:")

    print(
        df["embedding_bits"]
        .value_counts()
        .sort_index()
    )

    print("\nPercentages:")

    print(
        (
            df["embedding_bits"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nColumns:")
    print(
        list(df.columns)
    )

    print("=" * 70)


if __name__ == "__main__":
    main()