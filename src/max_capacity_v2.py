import os
import time
import cv2
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from preprocessing import rgb_to_ycbcr, ycbcr_to_rgb
from epvd import embed_pair_adaptive, extract_pair
from ai_adaptive import AdaptiveBitPredictor
from chaotic import generate_block_order


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest.pkl"
)

IMAGE_NAMES = [
    "lena.png",
    "baboon.jpeg",
    "car-house.jpeg",
    "jet.jpeg",
    "peppers.jpeg",
    "sailboat.jpeg"
]

SECRET_KEY = "AI_EPVD_2026"

BLOCK_SIZE = 8

# Large payload so the algorithm reaches its capacity.
PAYLOAD_BITS = 1_000_000

RNG_SEED = 2026


# ============================================================
# METRICS
# ============================================================

def calculate_mse(original, stego):

    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    return float(
        np.mean((original - stego) ** 2)
    )


def calculate_psnr(mse):

    if mse == 0:
        return float("inf")

    return float(
        10 * np.log10((255 ** 2) / mse)
    )


def calculate_ssim(original, stego):

    return float(
        ssim(
            original,
            stego,
            channel_axis=2,
            data_range=255
        )
    )


def calculate_ber(original_bits, extracted_bits):

    if len(original_bits) == 0:
        return 1.0

    if len(original_bits) != len(extracted_bits):
        return 1.0

    errors = sum(
        a != b
        for a, b in zip(
            original_bits,
            extracted_bits
        )
    )

    return errors / len(original_bits)


# ============================================================
# PAYLOAD
# ============================================================

def generate_payload():

    rng = np.random.default_rng(
        RNG_SEED
    )

    return ''.join(
        rng.choice(
            ['0', '1'],
            size=PAYLOAD_BITS
        )
    )


# ============================================================
# NATIVE EPVD CAPACITY
# ============================================================

def native_capacity(p1, p2):

    difference = abs(
        int(p2) - int(p1)
    )

    if difference <= 31:
        return 5

    elif difference <= 63:
        return 5

    elif difference <= 127:
        return 6

    else:
        return 8


# ============================================================
# STANDARD EPVD
# ============================================================

def embed_standard(
    y_channel,
    payload
):

    stego = y_channel.copy()

    bit_index = 0

    # Store every successful embedding operation.
    #
    # Each entry:
    # (row, col, number_of_bits)
    embedding_map = []

    height, width = stego.shape

    for i in range(
        0,
        height,
        BLOCK_SIZE
    ):

        for j in range(
            0,
            width,
            BLOCK_SIZE
        ):

            block = stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ].copy()

            for row in range(BLOCK_SIZE):

                for col in range(
                    0,
                    BLOCK_SIZE,
                    2
                ):

                    if bit_index >= len(payload):
                        break

                    p1 = int(
                        block[row, col]
                    )

                    p2 = int(
                        block[row, col + 1]
                    )

                    capacity = native_capacity(
                        p1,
                        p2
                    )

                    remaining = (
                        len(payload)
                        - bit_index
                    )

                    requested = min(
                        capacity,
                        remaining
                    )

                    secret = payload[
                        bit_index:
                        bit_index + requested
                    ]

                    new_p1, new_p2, actual, success = \
                        embed_pair_adaptive(
                            p1,
                            p2,
                            secret,
                            requested
                        )

                    if success and actual > 0:

                        block[row, col] = new_p1
                        block[row, col + 1] = new_p2

                        embedding_map.append(
                            (
                                i + row,
                                j + col,
                                actual
                            )
                        )

                        bit_index += actual

                if bit_index >= len(payload):
                    break

            stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ] = block

            if bit_index >= len(payload):
                break

        if bit_index >= len(payload):
            break

    return (
        stego,
        bit_index,
        embedding_map
    )


def extract_using_map(
    stego_y,
    embedding_map
):

    extracted = ""

    for row, col, number_of_bits in embedding_map:

        p1 = int(
            stego_y[row, col]
        )

        p2 = int(
            stego_y[row, col + 1]
        )

        bits, actual_bits = extract_pair(
            p1,
            p2,
            expected_bits=number_of_bits
        )

        extracted += bits

    return extracted


# ============================================================
# AI-ADAPTIVE EPVD
# ============================================================

def embed_ai(
    y_channel,
    payload,
    predictor
):

    stego = y_channel.copy()

    bit_index = 0

    embedding_map = []

    height, width = stego.shape

    # Batch-predict all blocks upfront
    all_blocks = []
    for bi in range(0, height, BLOCK_SIZE):
        for bj in range(0, width, BLOCK_SIZE):
            all_blocks.append(
                stego[bi:bi + BLOCK_SIZE, bj:bj + BLOCK_SIZE]
            )
    all_predictions = predictor.predict_blocks(all_blocks)
    block_idx = 0

    for i in range(
        0,
        height,
        BLOCK_SIZE
    ):

        for j in range(
            0,
            width,
            BLOCK_SIZE
        ):

            block = stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ].copy()

            predicted_bits = int(
                all_predictions[block_idx]
            )
            block_idx += 1

            for row in range(BLOCK_SIZE):

                for col in range(
                    0,
                    BLOCK_SIZE,
                    2
                ):

                    if bit_index >= len(payload):
                        break

                    p1 = int(
                        block[row, col]
                    )

                    p2 = int(
                        block[row, col + 1]
                    )

                    remaining = (
                        len(payload)
                        - bit_index
                    )

                    requested = min(
                        predicted_bits,
                        remaining
                    )

                    secret = payload[
                        bit_index:
                        bit_index + requested
                    ]

                    new_p1, new_p2, actual, success = \
                        embed_pair_adaptive(
                            p1,
                            p2,
                            secret,
                            predicted_bits
                        )

                    if success and actual > 0:

                        block[row, col] = new_p1
                        block[row, col + 1] = new_p2

                        embedding_map.append(
                            (
                                i + row,
                                j + col,
                                actual
                            )
                        )

                        bit_index += actual

                if bit_index >= len(payload):
                    break

            stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ] = block

            if bit_index >= len(payload):
                break

        if bit_index >= len(payload):
            break

    return (
        stego,
        bit_index,
        embedding_map
    )


# ============================================================
# AI + CHAOTIC EPVD
# ============================================================

def embed_secure(
    y_channel,
    payload,
    predictor,
    secret_key
):

    stego = y_channel.copy()

    height, width = stego.shape

    blocks_per_row = (
        width // BLOCK_SIZE
    )

    total_blocks = (
        (height // BLOCK_SIZE)
        * blocks_per_row
    )

    block_order = generate_block_order(
        total_blocks,
        secret_key
    )

    # Batch-predict all blocks upfront
    all_blocks = []
    for bi in range(total_blocks):
        br = bi // blocks_per_row
        bc = bi % blocks_per_row
        sy = br * BLOCK_SIZE
        sx = bc * BLOCK_SIZE
        all_blocks.append(
            stego[sy:sy + BLOCK_SIZE, sx:sx + BLOCK_SIZE]
        )
    all_predictions = predictor.predict_blocks(all_blocks)

    bit_index = 0

    embedding_map = []

    for block_id in block_order:

        if bit_index >= len(payload):
            break

        block_row = (
            block_id // blocks_per_row
        )

        block_col = (
            block_id % blocks_per_row
        )

        start_y = (
            block_row * BLOCK_SIZE
        )

        start_x = (
            block_col * BLOCK_SIZE
        )

        block = stego[
            start_y:start_y + BLOCK_SIZE,
            start_x:start_x + BLOCK_SIZE
        ].copy()

        predicted_bits = int(
            all_predictions[block_id]
        )

        for row in range(BLOCK_SIZE):

            for col in range(
                0,
                BLOCK_SIZE,
                2
            ):

                if bit_index >= len(payload):
                    break

                p1 = int(
                    block[row, col]
                )

                p2 = int(
                    block[row, col + 1]
                )

                remaining = (
                    len(payload)
                    - bit_index
                )

                requested = min(
                    predicted_bits,
                    remaining
                )

                secret = payload[
                    bit_index:
                    bit_index + requested
                ]

                new_p1, new_p2, actual, success = \
                    embed_pair_adaptive(
                        p1,
                        p2,
                        secret,
                        predicted_bits
                    )

                if success and actual > 0:

                    block[row, col] = new_p1
                    block[row, col + 1] = new_p2

                    embedding_map.append(
                        (
                            start_y + row,
                            start_x + col,
                            actual
                        )
                    )

                    bit_index += actual

        stego[
            start_y:start_y + BLOCK_SIZE,
            start_x:start_x + BLOCK_SIZE
        ] = block

    return (
        stego,
        bit_index,
        embedding_map
    )


# ============================================================
# IMAGE CONVERSION
# ============================================================

def make_stego_rgb(
    original_rgb,
    stego_y
):

    ycbcr = rgb_to_ycbcr(
        original_rgb
    )

    ycbcr[:, :, 0] = stego_y

    return ycbcr_to_rgb(
        ycbcr
    )


# ============================================================
# RUN ONE METHOD
# ============================================================

def run_method(
    image_name,
    method_name,
    original_rgb,
    cover_y,
    payload,
    predictor
):

    print(
        f"\n[{method_name}]"
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    start = time.perf_counter()

    if method_name == "Standard EPVD":

        stego_y, embedded_bits, embedding_map = \
            embed_standard(
                cover_y,
                payload
            )

    elif method_name == "AI-Adaptive EPVD":

        stego_y, embedded_bits, embedding_map = \
            embed_ai(
                cover_y,
                payload,
                predictor
            )

    else:

        stego_y, embedded_bits, embedding_map = \
            embed_secure(
                cover_y,
                payload,
                predictor,
                SECRET_KEY
            )

    embedding_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    start = time.perf_counter()

    extracted_bits = extract_using_map(
        stego_y,
        embedding_map
    )

    extraction_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    original_payload = payload[
        :embedded_bits
    ]

    ber = calculate_ber(
        original_payload,
        extracted_bits
    )

    stego_rgb = make_stego_rgb(
        original_rgb,
        stego_y
    )

    mse = calculate_mse(
        original_rgb,
        stego_rgb
    )

    psnr = calculate_psnr(
        mse
    )

    structural_similarity = calculate_ssim(
        original_rgb,
        stego_rgb
    )

    height, width = original_rgb.shape[:2]

    capacity_bpp = (
        embedded_bits /
        (height * width)
    )

    extraction_correct = (
        extracted_bits ==
        original_payload
    )

    print(
        f"Maximum bits : {embedded_bits}"
    )

    print(
        f"Capacity     : {capacity_bpp:.6f} bpp"
    )

    print(
        f"PSNR         : {psnr:.4f} dB"
    )

    print(
        f"SSIM         : {structural_similarity:.6f}"
    )

    print(
        f"BER          : {ber:.6f}"
    )

    print(
        f"Successful pairs : {len(embedding_map)}"
    )

    print(
        f"Extraction correct : {extraction_correct}"
    )

    return {
        "Image": image_name,
        "Method": method_name,
        "Maximum_Bits": embedded_bits,
        "Capacity_bpp": capacity_bpp,
        "MSE": mse,
        "PSNR_dB": psnr,
        "SSIM": structural_similarity,
        "BER": ber,
        "Successful_Pairs": len(embedding_map),
        "Embedding_Time_sec": embedding_time,
        "Extraction_Time_sec": extraction_time,
        "Extraction_Correct": extraction_correct
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    predictor = AdaptiveBitPredictor(
        MODEL_PATH
    )

    payload = generate_payload()

    print("=" * 70)
    print("CORRECTED MAXIMUM CAPACITY EVALUATION")
    print("=" * 70)

    print(
        f"Payload available: {len(payload)} bits"
    )

    results = []

    for image_name in IMAGE_NAMES:

        print("\n")
        print("-" * 70)
        print(
            f"IMAGE: {image_name}"
        )
        print("-" * 70)

        image_path = os.path.join(
            COVER_DIR,
            image_name
        )

        image_bgr = cv2.imread(
            image_path
        )

        if image_bgr is None:

            print(
                "Could not load image."
            )

            continue

        original_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        ycbcr = rgb_to_ycbcr(
            original_rgb
        )

        cover_y = ycbcr[
            :, :, 0
        ].copy()

        # Standard
        results.append(
            run_method(
                image_name,
                "Standard EPVD",
                original_rgb,
                cover_y,
                payload,
                predictor
            )
        )

        # AI
        results.append(
            run_method(
                image_name,
                "AI-Adaptive EPVD",
                original_rgb,
                cover_y,
                payload,
                predictor
            )
        )

        # AI + Chaotic
        results.append(
            run_method(
                image_name,
                "AI + Chaotic EPVD",
                original_rgb,
                cover_y,
                payload,
                predictor
            )
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "maximum_capacity_results_v2.csv"
    )

    df.to_csv(
        results_path,
        index=False
    )

    summary = df.groupby(
        "Method"
    )[
        [
            "Maximum_Bits",
            "Capacity_bpp",
            "MSE",
            "PSNR_dB",
            "SSIM",
            "BER",
            "Embedding_Time_sec",
            "Extraction_Time_sec"
        ]
    ].mean().reset_index()

    summary_path = os.path.join(
        RESULTS_DIR,
        "maximum_capacity_summary_v2.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL AVERAGE RESULTS")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        "Detailed results:"
    )

    print(
        results_path
    )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )

    print("=" * 70)


if __name__ == "__main__":
    main()