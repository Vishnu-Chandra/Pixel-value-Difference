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

# Payload rates to compare
PAYLOAD_RATES = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]

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
# PAYLOAD GENERATION
# ============================================================

def generate_payload(number_of_bits):

    rng = np.random.default_rng(
        RNG_SEED
    )

    return ''.join(
        rng.choice(
            ['0', '1'],
            size=number_of_bits
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

            if bit_index >= len(payload):
                break

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

            stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ] = block

        if bit_index >= len(payload):
            break

    return (
        stego,
        bit_index,
        embedding_map
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
    embedding_map = []

    height, width = stego.shape

    # Predict all blocks once
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

    block_index = 0

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
                break

            block = stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ].copy()

            predicted_bits = int(
                all_predictions[block_index]
            )

            block_index += 1

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

            stego[
                i:i + BLOCK_SIZE,
                j:j + BLOCK_SIZE
            ] = block

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

    # Predict every block once
    all_blocks = []

    for block_id in range(total_blocks):

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

        all_blocks.append(
            stego[
                start_y:start_y + BLOCK_SIZE,
                start_x:start_x + BLOCK_SIZE
            ]
        )

    all_predictions = predictor.predict_blocks(
        all_blocks
    )

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
                        requested
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
# EXTRACTION
# ============================================================

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
# STEGO CONVERSION
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

    results = []

    print("=" * 75)
    print("PAYLOAD-RATE SWEEP")
    print("=" * 75)

    print(
        "\nPayload rates:",
        PAYLOAD_RATES
    )

    for image_name in IMAGE_NAMES:

        print("\n")
        print("-" * 75)
        print(
            f"IMAGE: {image_name}"
        )
        print("-" * 75)

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

        height, width = cover_y.shape

        total_pixels = (
            height * width
        )

        for target_bpp in PAYLOAD_RATES:

            target_bits = int(
                target_bpp *
                total_pixels
            )

            payload = generate_payload(
                target_bits
            )

            print(
                f"\nTarget: {target_bpp:.2f} bpp "
                f"({target_bits} bits)"
            )

            # =================================================
            # STANDARD
            # =================================================

            start = time.perf_counter()

            standard_y, standard_bits, \
                standard_map = embed_standard(
                    cover_y,
                    payload
                )

            standard_embed_time = (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            standard_extracted = \
                extract_using_map(
                    standard_y,
                    standard_map
                )

            standard_extract_time = (
                time.perf_counter() - start
            )

            standard_original = payload[
                :standard_bits
            ]

            standard_ber = calculate_ber(
                standard_original,
                standard_extracted
            )

            standard_rgb = make_stego_rgb(
                original_rgb,
                standard_y
            )

            standard_mse = calculate_mse(
                original_rgb,
                standard_rgb
            )

            standard_psnr = calculate_psnr(
                standard_mse
            )

            standard_ssim = calculate_ssim(
                original_rgb,
                standard_rgb
            )

            results.append({
                "Image": image_name,
                "Target_bpp": target_bpp,
                "Method": "Standard EPVD",
                "Requested_Bits": target_bits,
                "Embedded_Bits": standard_bits,
                "Actual_bpp": standard_bits / total_pixels,
                "MSE": standard_mse,
                "PSNR_dB": standard_psnr,
                "SSIM": standard_ssim,
                "BER": standard_ber,
                "Embedding_Time_sec": standard_embed_time,
                "Extraction_Time_sec": standard_extract_time,
                "Extraction_Correct": standard_ber == 0.0
            })

            # =================================================
            # AI
            # =================================================

            start = time.perf_counter()

            ai_y, ai_bits, ai_map = embed_ai(
                cover_y,
                payload,
                predictor
            )

            ai_embed_time = (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            ai_extracted = extract_using_map(
                ai_y,
                ai_map
            )

            ai_extract_time = (
                time.perf_counter() - start
            )

            ai_original = payload[
                :ai_bits
            ]

            ai_ber = calculate_ber(
                ai_original,
                ai_extracted
            )

            ai_rgb = make_stego_rgb(
                original_rgb,
                ai_y
            )

            ai_mse = calculate_mse(
                original_rgb,
                ai_rgb
            )

            ai_psnr = calculate_psnr(
                ai_mse
            )

            ai_ssim = calculate_ssim(
                original_rgb,
                ai_rgb
            )

            results.append({
                "Image": image_name,
                "Target_bpp": target_bpp,
                "Method": "AI-Adaptive EPVD",
                "Requested_Bits": target_bits,
                "Embedded_Bits": ai_bits,
                "Actual_bpp": ai_bits / total_pixels,
                "MSE": ai_mse,
                "PSNR_dB": ai_psnr,
                "SSIM": ai_ssim,
                "BER": ai_ber,
                "Embedding_Time_sec": ai_embed_time,
                "Extraction_Time_sec": ai_extract_time,
                "Extraction_Correct": ai_ber == 0.0
            })

            # =================================================
            # AI + CHAOTIC
            # =================================================

            start = time.perf_counter()

            secure_y, secure_bits, \
                secure_map = embed_secure(
                    cover_y,
                    payload,
                    predictor,
                    SECRET_KEY
                )

            secure_embed_time = (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            secure_extracted = \
                extract_using_map(
                    secure_y,
                    secure_map
                )

            secure_extract_time = (
                time.perf_counter() - start
            )

            secure_original = payload[
                :secure_bits
            ]

            secure_ber = calculate_ber(
                secure_original,
                secure_extracted
            )

            secure_rgb = make_stego_rgb(
                original_rgb,
                secure_y
            )

            secure_mse = calculate_mse(
                original_rgb,
                secure_rgb
            )

            secure_psnr = calculate_psnr(
                secure_mse
            )

            secure_ssim = calculate_ssim(
                original_rgb,
                secure_rgb
            )

            results.append({
                "Image": image_name,
                "Target_bpp": target_bpp,
                "Method": "AI + Chaotic EPVD",
                "Requested_Bits": target_bits,
                "Embedded_Bits": secure_bits,
                "Actual_bpp": secure_bits / total_pixels,
                "MSE": secure_mse,
                "PSNR_dB": secure_psnr,
                "SSIM": secure_ssim,
                "BER": secure_ber,
                "Embedding_Time_sec": secure_embed_time,
                "Extraction_Time_sec": secure_extract_time,
                "Extraction_Correct": secure_ber == 0.0
            })

            print(
                f"  Standard: "
                f"{standard_bits / total_pixels:.3f} bpp | "
                f"PSNR {standard_psnr:.2f} | "
                f"BER {standard_ber:.4f}"
            )

            print(
                f"  AI:       "
                f"{ai_bits / total_pixels:.3f} bpp | "
                f"PSNR {ai_psnr:.2f} | "
                f"BER {ai_ber:.4f}"
            )

            print(
                f"  AI+Chaos: "
                f"{secure_bits / total_pixels:.3f} bpp | "
                f"PSNR {secure_psnr:.2f} | "
                f"BER {secure_ber:.4f}"
            )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    df = pd.DataFrame(
        results
    )

    detailed_path = os.path.join(
        RESULTS_DIR,
        "payload_sweep_results.csv"
    )

    df.to_csv(
        detailed_path,
        index=False
    )

    # ========================================================
    # AVERAGE BY METHOD + TARGET BPP
    # ========================================================

    summary = df.groupby(
        [
            "Method",
            "Target_bpp"
        ]
    )[
        [
            "Embedded_Bits",
            "Actual_bpp",
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
        "payload_sweep_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)
    print("PAYLOAD SWEEP SUMMARY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print("Saved:")
    print(detailed_path)
    print(summary_path)

    print("=" * 75)


if __name__ == "__main__":
    main()