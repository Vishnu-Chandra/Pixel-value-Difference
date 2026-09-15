import os
import sys
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
    PROJECT_ROOT,
    "data",
    "cover"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "embedding_level_analysis.csv"
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
# SIMULATE ONE PAIR
# =========================================================

def simulate_pair(
    p1,
    p2,
    bits,
    payload_value
):

    p1 = int(p1)
    p2 = int(p2)

    signed_difference = p2 - p1

    difference = abs(
        signed_difference
    )

    lower, upper = get_range(
        difference
    )

    if lower is None:
        return False, p1, p2

    capacity = get_capacity(
        lower,
        upper
    )

    if bits > capacity:
        return False, p1, p2

    if payload_value >= 2 ** bits:
        return False, p1, p2

    range_width = (
        upper - lower + 1
    )

    step = range_width / (
        2 ** bits
    )

    target_difference = int(
        lower
        + payload_value * step
        + step / 2
    )

    target_difference = min(
        target_difference,
        upper
    )

    if signed_difference < 0:
        target_difference = -target_difference

    delta = (
        target_difference
        - signed_difference
    )

    change_p1 = delta // 2
    change_p2 = delta - change_p1

    new_p1 = p1 - change_p1
    new_p2 = p2 + change_p2

    if (
        new_p1 < 0
        or new_p1 > 255
        or new_p2 < 0
        or new_p2 > 255
    ):
        return False, p1, p2

    return True, new_p1, new_p2


# =========================================================
# ANALYZE ONE BLOCK
# =========================================================

def analyze_block(block, bits):

    block = block.astype(
        np.int32
    )

    height, width = block.shape

    total_tests = 0
    successful_tests = 0

    total_squared_error = 0.0

    # Test every possible payload value
    payload_values = range(
        2 ** bits
    )

    for row in range(height):

        for col in range(
            0,
            width - 1,
            2
        ):

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

                    error = (
                        (new_p1 - p1) ** 2
                        + (new_p2 - p2) ** 2
                    )

                    total_squared_error += error

    success_ratio = (
        successful_tests / total_tests
        if total_tests > 0
        else 0.0
    )

    if successful_tests > 0:

        average_pair_error = (
            total_squared_error
            / successful_tests
        )

    else:

        average_pair_error = np.inf

    return (
        success_ratio,
        average_pair_error
    )


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze():

    print("----------------------------------------")
    print("Analyzing Embedding Levels")
    print("----------------------------------------")
    print()

    rows = []

    image_files = sorted([
        file
        for file in os.listdir(COVER_FOLDER)
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ])

    total_blocks = 0

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

        Y = ycbcr[:, :, 0]

        blocks = divide_into_blocks(
            Y,
            block_size=8
        )

        for block_id, block in enumerate(
            blocks
        ):

            result = {
                "image_name": image_name,
                "block_id": block_id
            }

            for bits in [2, 3, 4, 5]:

                success_ratio, error = analyze_block(
                    block,
                    bits
                )

                result[
                    f"success_{bits}"
                ] = success_ratio

                result[
                    f"mse_{bits}"
                ] = error

            rows.append(result)

            total_blocks += 1

        print(
            "  Blocks analyzed:",
            len(blocks)
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("----------------------------------------")
    print("Analysis completed!")
    print("----------------------------------------")

    print(
        "Total blocks:",
        len(df)
    )

    print()

    # =====================================================
    # STATISTICS
    # =====================================================

    print("Success-ratio statistics:")
    print()

    for bits in [2, 3, 4, 5]:

        column = f"success_{bits}"

        print(
            f"{bits}-bit:"
        )

        print(
            df[column].describe()
            .round(4)
        )

        print()

    print("----------------------------------------")
    print("MSE statistics")
    print("----------------------------------------")
    print()

    for bits in [2, 3, 4, 5]:

        column = f"mse_{bits}"

        valid = df[
            column
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        print(
            f"{bits}-bit:"
        )

        print(
            valid.describe()
            .round(4)
        )

        print()

    # =====================================================
    # SUCCESS THRESHOLD COUNTS
    # =====================================================

    print("----------------------------------------")
    print("Blocks meeting success thresholds")
    print("----------------------------------------")
    print()

    for threshold in [
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00
    ]:

        print(
            f"Threshold = {threshold}"
        )

        for bits in [2, 3, 4, 5]:

            count = (
                df[f"success_{bits}"]
                >= threshold
            ).sum()

            percentage = (
                count
                / len(df)
                * 100
            )

            print(
                f"  {bits}-bit: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

        print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    analyze()