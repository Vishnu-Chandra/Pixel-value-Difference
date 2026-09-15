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

# Large enough to fill the available capacity.
PAYLOAD_BITS = 1_000_000


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
        10 * np.log10(
            (255 ** 2) / mse
        )
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

    length = min(
        len(original_bits),
        len(extracted_bits)
    )

    if length == 0:
        return 1.0

    errors = sum(
        original_bits[i] != extracted_bits[i]
        for i in range(length)
    )

    return errors / length


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
    pairs_used = 0

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

            if bit_index >= len(payload):
                return (
                    stego,
                    bit_index,
                    pairs_used
                )

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

                    if success:

                        block[row, col] = new_p1
                        block[row, col + 1] = new_p2

                        bit_index += actual
                        pairs_used += 1

            stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ] = block

    return (
        stego,
        bit_index,
        pairs_used
    )


def extract_standard(
    cover_y,
    stego_y,
    expected_bits
):

    extracted = ""
    pairs_used = 0

    height, width = stego_y.shape

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

            if len(extracted) >= expected_bits:
                break

            cover_block = cover_y[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ]

            stego_block = stego_y[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ]

            for row in range(BLOCK_SIZE):

                for col in range(
                    0,
                    BLOCK_SIZE,
                    2
                ):

                    if len(extracted) >= expected_bits:
                        break

                    capacity = native_capacity(
                        cover_block[row, col],
                        cover_block[row, col + 1]
                    )

                    remaining = (
                        expected_bits
                        - len(extracted)
                    )

                    bits_to_extract = min(
                        capacity,
                        remaining
                    )

                    bits, actual_bits = extract_pair(
                        int(stego_block[row, col]),
                        int(stego_block[row, col + 1]),
                        expected_bits=bits_to_extract
                    )

                    extracted += bits
                    pairs_used += 1

            if len(extracted) >= expected_bits:
                break

    return (
        extracted[:expected_bits],
        pairs_used
    )


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
    pairs_used = 0
    blocks_used = 0

    height, width = stego.shape

    # Pre-compute ALL block predictions in one batch
    blocks_per_row = width // BLOCK_SIZE
    rows_of_blocks = height // BLOCK_SIZE

    all_blocks = []

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

            all_blocks.append(
                stego[
                    i:i + BLOCK_SIZE,
                    j:j + BLOCK_SIZE
                ]
            )

    all_predictions = predictor.predict_blocks(
        all_blocks
    )

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

            if bit_index >= len(payload):
                return (
                    stego,
                    bit_index,
                    pairs_used,
                    blocks_used
                )

            block = stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ].copy()

            predicted_bits = int(
                all_predictions[block_idx]
            )

            block_idx += 1

            block_embedded = False

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
                            requested
                        )

                    if success:

                        block[row, col] = new_p1
                        block[row, col + 1] = new_p2

                        bit_index += actual
                        pairs_used += 1
                        block_embedded = True

            stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ] = block

            if block_embedded:
                blocks_used += 1

    return (
        stego,
        bit_index,
        pairs_used,
        blocks_used
    )


def extract_ai(
    cover_y,
    stego_y,
    expected_bits,
    predictor
):

    extracted = ""
    pairs_used = 0
    blocks_used = 0

    height, width = stego_y.shape

    # Pre-compute ALL cover block predictions
    all_cover_blocks = []

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

            all_cover_blocks.append(
                cover_y[
                    i:i + BLOCK_SIZE,
                    j:j + BLOCK_SIZE
                ]
            )

    all_predictions = predictor.predict_blocks(
        all_cover_blocks
    )

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

            if len(extracted) >= expected_bits:
                break

            stego_block = stego_y[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ]

            predicted_bits = int(
                all_predictions[block_idx]
            )

            block_idx += 1

            block_extracted = False

            for row in range(BLOCK_SIZE):

                for col in range(
                    0,
                    BLOCK_SIZE,
                    2
                ):

                    if len(extracted) >= expected_bits:
                        break

                    remaining = (
                        expected_bits
                        - len(extracted)
                    )

                    bits_to_extract = min(
                        predicted_bits,
                        remaining
                    )

                    bits, actual_bits = extract_pair(
                        int(stego_block[row, col]),
                        int(stego_block[row, col + 1]),
                        expected_bits=bits_to_extract
                    )

                    extracted += bits
                    pairs_used += 1
                    block_extracted = True

            if block_extracted:
                blocks_used += 1

        if len(extracted) >= expected_bits:
            break

    return (
        extracted[:expected_bits],
        pairs_used,
        blocks_used
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

    blocks_per_row = width // BLOCK_SIZE

    total_blocks = (
        (height // BLOCK_SIZE)
        * blocks_per_row
    )

    block_order = generate_block_order(
        total_blocks,
        secret_key
    )

    # Pre-compute ALL block predictions in one batch
    all_blocks = []

    for bi in range(total_blocks):

        br = bi // blocks_per_row
        bc = bi % blocks_per_row

        sy = br * BLOCK_SIZE
        sx = bc * BLOCK_SIZE

        all_blocks.append(
            stego[
                sy:sy + BLOCK_SIZE,
                sx:sx + BLOCK_SIZE
            ]
        )

    all_predictions = predictor.predict_blocks(
        all_blocks
    )

    bit_index = 0
    pairs_used = 0
    blocks_used = 0

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

        block_embedded = False

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
                        requested
                    )

                if success:

                    block[row, col] = new_p1
                    block[row, col + 1] = new_p2

                    bit_index += actual
                    pairs_used += 1
                    block_embedded = True

        stego[
            start_y:start_y + BLOCK_SIZE,
            start_x:start_x + BLOCK_SIZE
        ] = block

        if block_embedded:
            blocks_used += 1

    return (
        stego,
        bit_index,
        pairs_used,
        blocks_used
    )


def extract_secure(
    cover_y,
    stego_y,
    expected_bits,
    predictor,
    secret_key
):

    height, width = stego_y.shape

    blocks_per_row = width // BLOCK_SIZE

    total_blocks = (
        (height // BLOCK_SIZE)
        * blocks_per_row
    )

    block_order = generate_block_order(
        total_blocks,
        secret_key
    )

    # Pre-compute ALL cover block predictions
    all_cover_blocks = []

    for bi in range(total_blocks):

        br = bi // blocks_per_row
        bc = bi % blocks_per_row

        sy = br * BLOCK_SIZE
        sx = bc * BLOCK_SIZE

        all_cover_blocks.append(
            cover_y[
                sy:sy + BLOCK_SIZE,
                sx:sx + BLOCK_SIZE
            ]
        )

    all_predictions = predictor.predict_blocks(
        all_cover_blocks
    )

    extracted = ""
    pairs_used = 0
    blocks_used = 0

    for block_id in block_order:

        if len(extracted) >= expected_bits:
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

        stego_block = stego_y[
            start_y:start_y + BLOCK_SIZE,
            start_x:start_x + BLOCK_SIZE
        ]

        predicted_bits = int(
            all_predictions[block_id]
        )

        block_extracted = False

        for row in range(BLOCK_SIZE):

            for col in range(
                0,
                BLOCK_SIZE,
                2
            ):

                if len(extracted) >= expected_bits:
                    break

                remaining = (
                    expected_bits
                    - len(extracted)
                )

                bits_to_extract = min(
                    predicted_bits,
                    remaining
                )

                bits, actual_bits = extract_pair(
                    int(stego_block[row, col]),
                    int(stego_block[row, col + 1]),
                    expected_bits=bits_to_extract
                )

                extracted += bits
                pairs_used += 1
                block_extracted = True

        if block_extracted:
            blocks_used += 1

    return (
        extracted[:expected_bits],
        pairs_used,
        blocks_used
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
# EVALUATION
# ============================================================

def evaluate_result(
    image_name,
    method,
    original_rgb,
    stego_y,
    embedded_bits,
    extracted_bits,
    pairs_used,
    blocks_used,
    embedding_time,
    extraction_time
):

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

    similarity = calculate_ssim(
        original_rgb,
        stego_rgb
    )

    # Generate deterministic payload again.
    rng = np.random.default_rng(2026)

    original_payload = ''.join(
        rng.choice(
            ['0', '1'],
            size=embedded_bits
        )
    )

    ber = calculate_ber(
        original_payload,
        extracted_bits
    )

    h, w = original_rgb.shape[:2]

    return {
        "Image": image_name,
        "Method": method,
        "Maximum_Bits": embedded_bits,
        "Capacity_bpp": embedded_bits / (h * w),
        "MSE": mse,
        "PSNR_dB": psnr,
        "SSIM": similarity,
        "BER": ber,
        "Pairs_Used": pairs_used,
        "Blocks_Used": blocks_used,
        "Embedding_Time_sec": embedding_time,
        "Extraction_Time_sec": extraction_time,
        "Extraction_Correct": ber == 0.0
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

    print("=" * 70)
    print("MAXIMUM CAPACITY EVALUATION")
    print("=" * 70)

    print(
        "\nPayload limit:",
        PAYLOAD_BITS,
        "bits"
    )

    results = []

    for image_name in IMAGE_NAMES:

        print("\n")
        print("-" * 70)
        print("IMAGE:", image_name)
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
                "ERROR: Could not load image."
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

        # ----------------------------------------------------
        # Generate deterministic large payload
        # ----------------------------------------------------

        rng = np.random.default_rng(
            2026
        )

        payload = ''.join(
            rng.choice(
                ['0', '1'],
                size=PAYLOAD_BITS
            )
        )

        # ====================================================
        # STANDARD
        # ====================================================

        print("\n[1/3] Standard EPVD")

        start = time.perf_counter()

        stego_y, embedded_bits, pairs = \
            embed_standard(
                cover_y,
                payload
            )

        embed_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        extracted, extract_pairs = \
            extract_standard(
                cover_y,
                stego_y,
                embedded_bits
            )

        extract_time = (
            time.perf_counter() - start
        )

        result = evaluate_result(
            image_name,
            "Standard EPVD",
            original_rgb,
            stego_y,
            embedded_bits,
            extracted,
            pairs,
            0,
            embed_time,
            extract_time
        )

        results.append(result)

        print(
            f"Maximum bits: {embedded_bits}"
        )

        print(
            f"Capacity: "
            f"{result['Capacity_bpp']:.6f} bpp"
        )

        print(
            f"PSNR: "
            f"{result['PSNR_dB']:.4f} dB"
        )

        print(
            f"SSIM: "
            f"{result['SSIM']:.6f}"
        )

        print(
            f"BER: "
            f"{result['BER']:.6f}"
        )

        # ====================================================
        # AI ADAPTIVE
        # ====================================================

        print("\n[2/3] AI-Adaptive EPVD")

        start = time.perf_counter()

        stego_y, embedded_bits, pairs, blocks = \
            embed_ai(
                cover_y,
                payload,
                predictor
            )

        embed_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        extracted, extract_pairs, \
            extract_blocks = extract_ai(
                cover_y,
                stego_y,
                embedded_bits,
                predictor
            )

        extract_time = (
            time.perf_counter() - start
        )

        result = evaluate_result(
            image_name,
            "AI-Adaptive EPVD",
            original_rgb,
            stego_y,
            embedded_bits,
            extracted,
            pairs,
            blocks,
            embed_time,
            extract_time
        )

        results.append(result)

        print(
            f"Maximum bits: {embedded_bits}"
        )

        print(
            f"Capacity: "
            f"{result['Capacity_bpp']:.6f} bpp"
        )

        print(
            f"PSNR: "
            f"{result['PSNR_dB']:.4f} dB"
        )

        print(
            f"SSIM: "
            f"{result['SSIM']:.6f}"
        )

        print(
            f"BER: "
            f"{result['BER']:.6f}"
        )

        # ====================================================
        # AI + CHAOTIC
        # ====================================================

        print("\n[3/3] AI + Chaotic EPVD")

        start = time.perf_counter()

        stego_y, embedded_bits, pairs, blocks = \
            embed_secure(
                cover_y,
                payload,
                predictor,
                SECRET_KEY
            )

        embed_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        extracted, extract_pairs, \
            extract_blocks = extract_secure(
                cover_y,
                stego_y,
                embedded_bits,
                predictor,
                SECRET_KEY
            )

        extract_time = (
            time.perf_counter() - start
        )

        result = evaluate_result(
            image_name,
            "AI + Chaotic EPVD",
            original_rgb,
            stego_y,
            embedded_bits,
            extracted,
            pairs,
            blocks,
            embed_time,
            extract_time
        )

        results.append(result)

        print(
            f"Maximum bits: {embedded_bits}"
        )

        print(
            f"Capacity: "
            f"{result['Capacity_bpp']:.6f} bpp"
        )

        print(
            f"PSNR: "
            f"{result['PSNR_dB']:.4f} dB"
        )

        print(
            f"SSIM: "
            f"{result['SSIM']:.6f}"
        )

        print(
            f"BER: "
            f"{result['BER']:.6f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    df = pd.DataFrame(
        results
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "maximum_capacity_results.csv"
    )

    df.to_csv(
        output_path,
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
        "maximum_capacity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("AVERAGE MAXIMUM CAPACITY RESULTS")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nResults saved:")
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()