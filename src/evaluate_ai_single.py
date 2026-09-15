import os
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim

from preprocessing import divide_into_blocks
from ai_adaptive import AdaptiveBitPredictor
from epvd import embed_pair_adaptive, extract_pair


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover",
    "lena.png"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "evaluation_ai_lena.png"
)

MESSAGE = "AI EPVD Evaluation Message"


# ============================================================
# Text / Bits
# ============================================================

def text_to_bits(text):
    return "".join(
        format(ord(c), "08b")
        for c in text
    )


# ============================================================
# Metrics
# ============================================================

def calculate_mse(original, stego):
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    return np.mean(
        (original - stego) ** 2
    )


def calculate_psnr(original, stego):

    mse = calculate_mse(
        original,
        stego
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10(
        (255 ** 2) / mse
    )


def calculate_ssim(original, stego):

    return ssim(
        original,
        stego,
        channel_axis=2,
        data_range=255
    )


def calculate_ber(original_bits, extracted_bits):

    length = min(
        len(original_bits),
        len(extracted_bits)
    )

    if length == 0:
        return 1.0

    errors = sum(
        a != b
        for a, b in zip(
            original_bits[:length],
            extracted_bits[:length]
        )
    )

    return errors / length


# ============================================================
# Embedding
# ============================================================

def embed_image(
    cover_y,
    secret_bits,
    predictor
):

    stego_y = cover_y.copy()

    height, width = cover_y.shape

    blocks = divide_into_blocks(
        cover_y,
        block_size=8
    )

    predictions = predictor.predict_blocks(
        blocks
    )

    embedding_map = []

    bit_index = 0

    block_index = 0

    pairs_attempted = 0
    successful_pairs = 0
    failed_pairs = 0

    for block_row in range(
        0,
        height,
        8
    ):

        for block_col in range(
            0,
            width,
            8
        ):

            if bit_index >= len(secret_bits):
                break

            block = stego_y[
                block_row:block_row + 8,
                block_col:block_col + 8
            ].copy()

            requested_bits = int(
                predictions[block_index]
            )

            for row in range(8):

                for col in range(
                    0,
                    8,
                    2
                ):

                    if bit_index >= len(secret_bits):
                        break

                    pairs_attempted += 1

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

                    requested_for_pair = min(
                        requested_bits,
                        remaining
                    )

                    payload = secret_bits[
                        bit_index:
                        bit_index + requested_for_pair
                    ]

                    (
                        new_p1,
                        new_p2,
                        actual_bits,
                        success
                    ) = embed_pair_adaptive(
                        p1,
                        p2,
                        payload,
                        requested_bits
                    )

                    if success and actual_bits > 0:

                        block[row, col] = new_p1
                        block[row, col + 1] = new_p2

                        embedded_bits = secret_bits[
                            bit_index:
                            bit_index + actual_bits
                        ]

                        embedding_map.append({
                            "row": block_row + row,
                            "col": block_col + col,
                            "bits": actual_bits,
                            "payload": embedded_bits
                        })

                        bit_index += actual_bits

                        successful_pairs += 1

                    else:
                        failed_pairs += 1

            stego_y[
                block_row:block_row + 8,
                block_col:block_col + 8
            ] = block

            block_index += 1

        if bit_index >= len(secret_bits):
            break

    return (
        stego_y,
        embedding_map,
        bit_index,
        pairs_attempted,
        successful_pairs,
        failed_pairs
    )


# ============================================================
# Extraction
# ============================================================

def extract_using_map(
    stego_y,
    embedding_map
):

    extracted_bits = ""

    for item in embedding_map:

        row = item["row"]
        col = item["col"]
        bits = item["bits"]

        p1 = int(
            stego_y[row, col]
        )

        p2 = int(
            stego_y[row, col + 1]
        )

        extracted, actual_bits = extract_pair(
            p1,
            p2,
            bits
        )

        extracted_bits += extracted[
            :actual_bits
        ]

    return extracted_bits


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("AI-ADAPTIVE EPVD SINGLE-IMAGE EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Load cover
# ------------------------------------------------------------

cover_bgr = cv2.imread(
    COVER_PATH
)

if cover_bgr is None:
    raise FileNotFoundError(
        COVER_PATH
    )

cover_rgb = cv2.cvtColor(
    cover_bgr,
    cv2.COLOR_BGR2RGB
)

cover_ycrcb = cv2.cvtColor(
    cover_rgb,
    cv2.COLOR_RGB2YCrCb
)

cover_y = cover_ycrcb[:, :, 0]


# ------------------------------------------------------------
# Secret
# ------------------------------------------------------------

secret_bits = text_to_bits(
    MESSAGE
)

print(
    f"\nMessage       : {MESSAGE}"
)

print(
    f"Secret bits   : {len(secret_bits)}"
)


# ------------------------------------------------------------
# AI embedding
# ------------------------------------------------------------

predictor = AdaptiveBitPredictor()

(
    stego_y,
    embedding_map,
    bits_embedded,
    pairs_attempted,
    successful_pairs,
    failed_pairs
) = embed_image(
    cover_y,
    secret_bits,
    predictor
)


# ------------------------------------------------------------
# Reconstruct stego image
# ------------------------------------------------------------

stego_ycrcb = cover_ycrcb.copy()

stego_ycrcb[:, :, 0] = stego_y

stego_rgb = cv2.cvtColor(
    stego_ycrcb,
    cv2.COLOR_YCrCb2RGB
)

stego_bgr = cv2.cvtColor(
    stego_rgb,
    cv2.COLOR_RGB2BGR
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_PATH,
    stego_bgr
)


# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------

extracted_bits = extract_using_map(
    stego_y,
    embedding_map
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

mse = calculate_mse(
    cover_rgb,
    stego_rgb
)

psnr = calculate_psnr(
    cover_rgb,
    stego_rgb
)

ssim_value = calculate_ssim(
    cover_rgb,
    stego_rgb
)

ber = calculate_ber(
    secret_bits,
    extracted_bits
)

extraction_correct = (
    extracted_bits == secret_bits
)

actual_bpp = (
    bits_embedded
    /
    (cover_y.shape[0] * cover_y.shape[1])
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"MSE                 : {mse:.6f}"
)

print(
    f"PSNR                : {psnr:.4f} dB"
)

print(
    f"SSIM                : {ssim_value:.6f}"
)

print(
    f"BER                 : {ber:.6f}"
)

print(
    f"Bits embedded       : {bits_embedded}"
)

print(
    f"Actual payload      : {actual_bpp:.6f} bpp"
)

print(
    f"Pairs attempted     : {pairs_attempted}"
)

print(
    f"Successful pairs    : {successful_pairs}"
)

print(
    f"Failed pairs        : {failed_pairs}"
)

print(
    f"Embedding map size  : {len(embedding_map)}"
)

print(
    f"Extracted bits      : {len(extracted_bits)}"
)

print(
    f"Extraction correct  : {extraction_correct}"
)

print(
    f"\nSaved stego image:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)