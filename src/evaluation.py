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

SECRET_MESSAGE = "Hello Secure AI EPVD"

SECRET_BITS = ''.join(
    format(ord(c), '08b')
    for c in SECRET_MESSAGE
)

SECRET_KEY = "AI_EPVD_2026"

BLOCK_SIZE = 8


# ============================================================
# BASIC UTILITIES
# ============================================================

def load_rgb(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def save_rgb(image, path):

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        bgr
    )


def text_to_bits(text):

    return ''.join(
        format(ord(c), '08b')
        for c in text
    )


def bits_to_text(bits):

    chars = []

    for i in range(0, len(bits), 8):

        byte = bits[i:i + 8]

        if len(byte) < 8:
            break

        chars.append(
            chr(int(byte, 2))
        )

    return ''.join(chars)


def calculate_mse(original, stego):

    original = original.astype(
        np.float64
    )

    stego = stego.astype(
        np.float64
    )

    return float(
        np.mean(
            (original - stego) ** 2
        )
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
# STANDARD EPVD EMBEDDING
# ============================================================

def embed_standard(
    y_channel,
    secret_bits
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

            if bit_index >= len(secret_bits):
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

                    if bit_index >= len(secret_bits):
                        break

                    p1 = int(
                        block[row, col]
                    )

                    p2 = int(
                        block[row, col + 1]
                    )

                    difference = abs(
                        p2 - p1
                    )

                    if difference <= 31:
                        capacity = 5
                    elif difference <= 63:
                        capacity = 5
                    elif difference <= 127:
                        capacity = 6
                    else:
                        capacity = 7

                    remaining = (
                        len(secret_bits)
                        - bit_index
                    )

                    requested = min(
                        capacity,
                        remaining
                    )

                    payload = secret_bits[
                        bit_index:
                        bit_index + requested
                    ]

                    new_p1, new_p2, actual, success = \
                        embed_pair_adaptive(
                            p1,
                            p2,
                            payload,
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


# ============================================================
# STANDARD EPVD EXTRACTION
# ============================================================

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

                    original_difference = abs(
                        int(cover_block[row, col + 1])
                        - int(cover_block[row, col])
                    )

                    if original_difference <= 31:
                        native_bits = 5
                    elif original_difference <= 63:
                        native_bits = 5
                    elif original_difference <= 127:
                        native_bits = 6
                    else:
                        native_bits = 7

                    remaining = (
                        expected_bits
                        - len(extracted)
                    )

                    bits_to_extract = min(
                        native_bits,
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
# AI-ADAPTIVE EMBEDDING
# ============================================================

def embed_ai(
    y_channel,
    secret_bits,
    predictor
):

    stego = y_channel.copy()

    bit_index = 0
    pairs_used = 0
    blocks_used = 0

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

            if bit_index >= len(secret_bits):
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

            requested_bits = predictor.predict_block(
                block
            )

            block_embedded = False

            for row in range(BLOCK_SIZE):

                for col in range(
                    0,
                    BLOCK_SIZE,
                    2
                ):

                    if bit_index >= len(secret_bits):
                        break

                    p1 = int(
                        block[row, col]
                    )

                    p2 = int(
                        block[row, col + 1]
                    )

                    remaining = (
                        len(secret_bits)
                        - bit_index
                    )

                    requested = min(
                        requested_bits,
                        remaining
                    )

                    payload = secret_bits[
                        bit_index:
                        bit_index + requested
                    ]

                    new_p1, new_p2, actual, success = \
                        embed_pair_adaptive(
                            p1,
                            p2,
                            payload,
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


# ============================================================
# AI-ADAPTIVE EXTRACTION
# ============================================================

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

            requested_bits = predictor.predict_block(
                cover_block
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
                        requested_bits,
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
# AI + CHAOTIC EMBEDDING
# ============================================================

def embed_secure(
    y_channel,
    secret_bits,
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

    bit_index = 0
    pairs_used = 0
    blocks_used = 0

    for block_id in block_order:

        if bit_index >= len(secret_bits):
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

        requested_bits = predictor.predict_block(
            block
        )

        block_embedded = False

        for row in range(BLOCK_SIZE):

            for col in range(
                0,
                BLOCK_SIZE,
                2
            ):

                if bit_index >= len(secret_bits):
                    break

                p1 = int(
                    block[row, col]
                )

                p2 = int(
                    block[row, col + 1]
                )

                remaining = (
                    len(secret_bits)
                    - bit_index
                )

                requested = min(
                    requested_bits,
                    remaining
                )

                payload = secret_bits[
                    bit_index:
                    bit_index + requested
                ]

                new_p1, new_p2, actual, success = \
                    embed_pair_adaptive(
                        p1,
                        p2,
                        payload,
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


# ============================================================
# AI + CHAOTIC EXTRACTION
# ============================================================

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

        cover_block = cover_y[
            start_y:start_y + BLOCK_SIZE,
            start_x:start_x + BLOCK_SIZE
        ]

        stego_block = stego_y[
            start_y:start_y + BLOCK_SIZE,
            start_x:start_x + BLOCK_SIZE
        ]

        requested_bits = predictor.predict_block(
            cover_block
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
                    requested_bits,
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
# CONVERT Y CHANNEL BACK TO RGB
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
# MAIN EVALUATION
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
    print("AI-EPVD COMPLETE PERFORMANCE EVALUATION")
    print("=" * 70)

    print(
        f"\nSecret message: {SECRET_MESSAGE}"
    )

    print(
        f"Secret length: {len(SECRET_BITS)} bits"
    )

    all_results = []

    for image_name in IMAGE_NAMES:

        print("\n")
        print("-" * 70)
        print(f"IMAGE: {image_name}")
        print("-" * 70)

        cover_path = os.path.join(
            COVER_DIR,
            image_name
        )

        original_rgb = load_rgb(
            cover_path
        )

        ycbcr = rgb_to_ycbcr(
            original_rgb
        )

        cover_y = ycbcr[:, :, 0].copy()

        # ====================================================
        # STANDARD EPVD
        # ====================================================

        print("\n[1/3] Standard EPVD")

        start = time.perf_counter()

        standard_y, standard_bits, standard_pairs = \
            embed_standard(
                cover_y,
                SECRET_BITS
            )

        embedding_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        standard_extracted, standard_extract_pairs = \
            extract_standard(
                cover_y,
                standard_y,
                standard_bits
            )

        extraction_time = (
            time.perf_counter() - start
        )

        standard_rgb = make_stego_rgb(
            original_rgb,
            standard_y
        )

        mse = calculate_mse(
            original_rgb,
            standard_rgb
        )

        psnr = calculate_psnr(mse)

        similarity = calculate_ssim(
            original_rgb,
            standard_rgb
        )

        ber = calculate_ber(
            SECRET_BITS[:standard_bits],
            standard_extracted
        )

        message = bits_to_text(
            standard_extracted
        )

        correct = (
            message == SECRET_MESSAGE
        )

        h, w = original_rgb.shape[:2]

        all_results.append({
            "Image": image_name,
            "Method": "Standard EPVD",
            "MSE": mse,
            "PSNR_dB": psnr,
            "SSIM": similarity,
            "Embedded_Bits": standard_bits,
            "Capacity_bpp": standard_bits / (h * w),
            "BER": ber,
            "Pairs_Used": standard_pairs,
            "Blocks_Used": 0,
            "Embedding_Time_sec": embedding_time,
            "Extraction_Time_sec": extraction_time,
            "Extraction_Correct": correct
        })

        print(
            f"Embedded: {standard_bits} bits"
        )

        print(
            f"Extracted: {len(standard_extracted)} bits"
        )

        print(
            f"BER: {ber:.6f}"
        )

        print(
            f"PSNR: {psnr:.4f} dB"
        )

        print(
            f"SSIM: {similarity:.6f}"
        )

        print(
            f"Message correct: {correct}"
        )

        # ====================================================
        # AI ADAPTIVE EPVD
        # ====================================================

        print("\n[2/3] AI-Adaptive EPVD")

        start = time.perf_counter()

        ai_y, ai_bits, ai_pairs, ai_blocks = \
            embed_ai(
                cover_y,
                SECRET_BITS,
                predictor
            )

        embedding_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        ai_extracted, ai_extract_pairs, \
            ai_extract_blocks = extract_ai(
                cover_y,
                ai_y,
                ai_bits,
                predictor
            )

        extraction_time = (
            time.perf_counter() - start
        )

        ai_rgb = make_stego_rgb(
            original_rgb,
            ai_y
        )

        mse = calculate_mse(
            original_rgb,
            ai_rgb
        )

        psnr = calculate_psnr(mse)

        similarity = calculate_ssim(
            original_rgb,
            ai_rgb
        )

        ber = calculate_ber(
            SECRET_BITS[:ai_bits],
            ai_extracted
        )

        message = bits_to_text(
            ai_extracted
        )

        correct = (
            message == SECRET_MESSAGE
        )

        all_results.append({
            "Image": image_name,
            "Method": "AI-Adaptive EPVD",
            "MSE": mse,
            "PSNR_dB": psnr,
            "SSIM": similarity,
            "Embedded_Bits": ai_bits,
            "Capacity_bpp": ai_bits / (h * w),
            "BER": ber,
            "Pairs_Used": ai_pairs,
            "Blocks_Used": ai_blocks,
            "Embedding_Time_sec": embedding_time,
            "Extraction_Time_sec": extraction_time,
            "Extraction_Correct": correct
        })

        print(
            f"Embedded: {ai_bits} bits"
        )

        print(
            f"Extracted: {len(ai_extracted)} bits"
        )

        print(
            f"BER: {ber:.6f}"
        )

        print(
            f"PSNR: {psnr:.4f} dB"
        )

        print(
            f"SSIM: {similarity:.6f}"
        )

        print(
            f"Message correct: {correct}"
        )

        # ====================================================
        # AI + CHAOTIC EPVD
        # ====================================================

        print("\n[3/3] AI + Chaotic EPVD")

        start = time.perf_counter()

        secure_y, secure_bits, \
            secure_pairs, secure_blocks = \
            embed_secure(
                cover_y,
                SECRET_BITS,
                predictor,
                SECRET_KEY
            )

        embedding_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        secure_extracted, secure_extract_pairs, \
            secure_extract_blocks = \
            extract_secure(
                cover_y,
                secure_y,
                secure_bits,
                predictor,
                SECRET_KEY
            )

        extraction_time = (
            time.perf_counter() - start
        )

        secure_rgb = make_stego_rgb(
            original_rgb,
            secure_y
        )

        mse = calculate_mse(
            original_rgb,
            secure_rgb
        )

        psnr = calculate_psnr(mse)

        similarity = calculate_ssim(
            original_rgb,
            secure_rgb
        )

        ber = calculate_ber(
            SECRET_BITS[:secure_bits],
            secure_extracted
        )

        message = bits_to_text(
            secure_extracted
        )

        correct = (
            message == SECRET_MESSAGE
        )

        all_results.append({
            "Image": image_name,
            "Method": "AI + Chaotic EPVD",
            "MSE": mse,
            "PSNR_dB": psnr,
            "SSIM": similarity,
            "Embedded_Bits": secure_bits,
            "Capacity_bpp": secure_bits / (h * w),
            "BER": ber,
            "Pairs_Used": secure_pairs,
            "Blocks_Used": secure_blocks,
            "Embedding_Time_sec": embedding_time,
            "Extraction_Time_sec": extraction_time,
            "Extraction_Correct": correct
        })

        print(
            f"Embedded: {secure_bits} bits"
        )

        print(
            f"Extracted: {len(secure_extracted)} bits"
        )

        print(
            f"BER: {ber:.6f}"
        )

        print(
            f"PSNR: {psnr:.4f} dB"
        )

        print(
            f"SSIM: {similarity:.6f}"
        )

        print(
            f"Message correct: {correct}"
        )

        # Save final stego images
        save_rgb(
            standard_rgb,
            os.path.join(
                RESULTS_DIR,
                f"{os.path.splitext(image_name)[0]}_standard.png"
            )
        )

        save_rgb(
            ai_rgb,
            os.path.join(
                RESULTS_DIR,
                f"{os.path.splitext(image_name)[0]}_ai.png"
            )
        )

        save_rgb(
            secure_rgb,
            os.path.join(
                RESULTS_DIR,
                f"{os.path.splitext(image_name)[0]}_secure.png"
            )
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    df = pd.DataFrame(
        all_results
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "evaluation_results.csv"
    )

    df.to_csv(
        results_path,
        index=False
    )

    # Average results
    summary = df.groupby(
        "Method"
    )[
        [
            "MSE",
            "PSNR_dB",
            "SSIM",
            "Embedded_Bits",
            "Capacity_bpp",
            "BER",
            "Embedding_Time_sec",
            "Extraction_Time_sec"
        ]
    ].mean().reset_index()

    summary_path = os.path.join(
        RESULTS_DIR,
        "evaluation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

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
    print("Files generated:")
    print(
        f"  {results_path}"
    )

    print(
        f"  {summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()