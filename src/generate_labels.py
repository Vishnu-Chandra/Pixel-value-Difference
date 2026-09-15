import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_image, rgb_to_ycbcr, divide_into_blocks


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_FOLDER = os.path.join(
    PROJECT_ROOT, "data", "cover"
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT, "data", "features.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "labeled_features.csv"
)

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


# =========================================================
# EPVD RANGES
# =========================================================

RANGES = [
    (0, 31),
    (32, 63),
    (64, 127),
    (128, 255)
]


def get_range(difference):

    for lower, upper in RANGES:

        if lower <= difference <= upper:
            return lower, upper

    return None, None


def get_capacity(lower, upper):

    width = upper - lower + 1

    return int(
        np.floor(np.log2(width))
    )


# =========================================================
# EMBED ONE PAYLOAD VALUE
# =========================================================

def simulate_pair(p1, p2, bits, payload_value):

    p1 = int(p1)
    p2 = int(p2)

    signed_difference = p2 - p1
    difference = abs(signed_difference)

    lower, upper = get_range(difference)

    if lower is None:
        return False, p1, p2

    capacity = get_capacity(
        lower,
        upper
    )

    # Requested embedding depth must be
    # supported by the EPVD range.
    if bits > capacity:
        return False, p1, p2

    # Payload value must fit inside requested bits.
    if payload_value < 0 or payload_value >= 2 ** bits:
        return False, p1, p2

    # Map payload value into the EPVD range.
    range_width = upper - lower + 1

    step = range_width / (2 ** bits)

    target_difference = int(
        lower
        + payload_value * step
        + step / 2
    )

    target_difference = min(
        target_difference,
        upper
    )

    # Preserve original direction.
    if signed_difference < 0:
        target_difference = -target_difference

    # Difference adjustment.
    delta = (
        target_difference
        - signed_difference
    )

    change_p1 = delta // 2
    change_p2 = delta - change_p1

    new_p1 = p1 - change_p1
    new_p2 = p2 + change_p2

    # Boundary validation.
    if (
        new_p1 < 0
        or new_p1 > 255
        or new_p2 < 0
        or new_p2 > 255
    ):
        return False, p1, p2

    return True, new_p1, new_p2


# =========================================================
# TEST ALL PAYLOAD VALUES ON A BLOCK
# =========================================================

def evaluate_bit_level(block, bits):

    block = block.astype(np.int32)

    height, width = block.shape

    total_tests = 0
    successful_tests = 0

    total_mse = 0.0

    # We test every possible payload value for the
    # requested number of bits.
    payload_values = range(
        2 ** bits
    )

    for row in range(height):

        for col in range(0, width - 1, 2):

            p1 = block[row, col]
            p2 = block[row, col + 1]

            for payload_value in payload_values:

                total_tests += 1

                success, new_p1, new_p2 = simulate_pair(
                    p1,
                    p2,
                    bits,
                    payload_value
                )

                if success:

                    successful_tests += 1

                    distortion = (
                        (float(new_p1) - float(p1)) ** 2
                        +
                        (float(new_p2) - float(p2)) ** 2
                    )

                    total_mse += distortion

    if total_tests == 0:

        return 0.0, float("inf")

    success_ratio = (
        successful_tests / total_tests
    )

    if successful_tests > 0:

        average_distortion = (
            total_mse / successful_tests
        )

    else:

        average_distortion = float("inf")

    return success_ratio, average_distortion


# =========================================================
# DETERMINE ADAPTIVE LABEL
# =========================================================

def determine_label(block):

    evaluation = {}

    for bits in [2, 3, 4, 5]:

        success_ratio, distortion = evaluate_bit_level(
            block,
            bits
        )

        evaluation[bits] = {
            "success_ratio": success_ratio,
            "distortion": distortion
        }

    # -----------------------------------------------------
    # Select the highest capacity that is sufficiently safe.
    # -----------------------------------------------------

    # 5-bit suitability
    if (
        evaluation[5]["success_ratio"] >= 0.90
    ):
        return 5

    # 4-bit suitability
    if (
        evaluation[4]["success_ratio"] >= 0.90
    ):
        return 4

    # 3-bit suitability
    if (
        evaluation[3]["success_ratio"] >= 0.90
    ):
        return 3

    # Conservative fallback
    return 2


# =========================================================
# MAIN
# =========================================================

def generate_labels():

    print("----------------------------------------")
    print("Generating Adaptive Embedding Labels")
    print("----------------------------------------")
    print()

    features_df = pd.read_csv(
        FEATURE_FILE
    )

    labels = []

    image_files = sorted([
        file
        for file in os.listdir(COVER_FOLDER)
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ])

    for image_name in image_files:

        print(
            "Processing:",
            image_name
        )

        image_path = os.path.join(
            COVER_FOLDER,
            image_name
        )

        image = load_image(
            image_path
        )

        ycbcr = rgb_to_ycbcr(
            image
        )

        # OpenCV stores the channels as:
        # Y, Cr, Cb
        Y = ycbcr[:, :, 0]

        blocks = divide_into_blocks(
            Y,
            block_size=8
        )

        print(
            "  Blocks:",
            len(blocks)
        )

        image_labels = []

        for block in blocks:

            label = determine_label(
                block
            )

            labels.append(
                label
            )

            image_labels.append(
                label
            )

        unique, counts = np.unique(
            image_labels,
            return_counts=True
        )

        print(
            "  Label distribution:"
        )

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"    {label} bits: {count}"
            )

        print()

    # -----------------------------------------------------
    # Verify feature/label alignment
    # -----------------------------------------------------

    if len(labels) != len(features_df):

        raise ValueError(
            f"Feature count "
            f"({len(features_df)}) "
            f"does not match label count "
            f"({len(labels)})"
        )

    features_df[
        "embedding_bits"
    ] = labels

    # -----------------------------------------------------
    # Save labeled dataset
    # -----------------------------------------------------

    features_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("----------------------------------------")
    print("Label generation completed!")
    print("----------------------------------------")

    print(
        "Total blocks:",
        len(features_df)
    )

    print()

    print(
        "Overall label distribution:"
    )

    distribution = (
        features_df[
            "embedding_bits"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        distribution
    )

    print()

    print(
        "Percentage distribution:"
    )

    percentages = (
        features_df[
            "embedding_bits"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        percentages.round(2)
    )

    print()

    print(
        "Dataset shape:",
        features_df.shape
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    generate_labels()