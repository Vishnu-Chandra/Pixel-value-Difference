import os
import cv2
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from preprocessing import divide_into_blocks
from ai_adaptive import AdaptiveBitPredictor
from epvd import embed_pair_adaptive, extract_pair


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


IMAGE_NAMES = [
    "lena.png",
    "baboon.jpeg",
    "car-house.jpeg",
    "jet.jpeg",
    "peppers.jpeg",
    "sailboat.jpeg"
]

PAYLOAD_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]


# ============================================================
# Metrics
# ============================================================

def calculate_mse(
    original,
    stego
):

    original = original.astype(
        np.float64
    )

    stego = stego.astype(
        np.float64
    )

    return np.mean(
        (original - stego) ** 2
    )


def calculate_psnr(
    original,
    stego
):

    mse = calculate_mse(
        original,
        stego
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10(
        (255 ** 2) / mse
    )


def calculate_ssim(
    original,
    stego
):

    return ssim(
        original,
        stego,
        channel_axis=2,
        data_range=255
    )


def calculate_ber(
    original_bits,
    extracted_bits
):

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

    # Missing bits count as errors
    errors += abs(
        len(original_bits)
        -
        len(extracted_bits)
    )

    return errors / len(
        original_bits
    )


# ============================================================
# Deterministic Payload
# ============================================================

def generate_payload(length):

    rng = np.random.default_rng(
        2026
    )

    return "".join(
        rng.choice(
            ["0", "1"],
            size=length
        )
    )


# ============================================================
# AI Embedding
# ============================================================

def embed_ai(
    cover_y,
    secret_bits,
    predictions
):

    stego_y = cover_y.copy()

    height, width = cover_y.shape

    bit_index = 0

    embedding_map = []

    for block_index, prediction in enumerate(
        predictions
    ):

        if bit_index >= len(secret_bits):
            break

        block_row = (
            block_index
            // (width // 8)
        ) * 8

        block_col = (
            block_index
            % (width // 8)
        ) * 8

        block = stego_y[
            block_row:block_row + 8,
            block_col:block_col + 8
        ].copy()

        requested_bits = int(
            prediction
        )

        for row in range(8):

            for col in range(
                0,
                8,
                2
            ):

                if bit_index >= len(secret_bits):
                    break

                remaining = (
                    len(secret_bits)
                    -
                    bit_index
                )

                requested_for_pair = min(
                    requested_bits,
                    remaining
                )

                payload = secret_bits[
                    bit_index:
                    bit_index
                    +
                    requested_for_pair
                ]

                p1 = int(
                    block[row, col]
                )

                p2 = int(
                    block[row, col + 1]
                )

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

                    embedding_map.append({
                        "row": block_row + row,
                        "col": block_col + col,
                        "bits": actual_bits
                    })

                    bit_index += actual_bits

        stego_y[
            block_row:block_row + 8,
            block_col:block_col + 8
        ] = block

    return (
        stego_y,
        embedding_map,
        bit_index
    )


# ============================================================
# Extraction
# ============================================================

def extract_ai(
    stego_y,
    embedding_map
):

    extracted = []

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

        payload, actual_bits = extract_pair(
            p1,
            p2,
            bits
        )

        extracted.append(
            payload[:actual_bits]
        )

    return "".join(extracted)


# ============================================================
# Main Experiment
# ============================================================

print("=" * 70)
print("AI-ADAPTIVE EPVD PAYLOAD EXPERIMENT")
print("=" * 70)


predictor = AdaptiveBitPredictor()

results = []


for image_name in IMAGE_NAMES:

    print(
        f"\nProcessing: {image_name}"
    )

    cover_path = os.path.join(
        COVER_DIR,
        image_name
    )

    cover_bgr = cv2.imread(
        cover_path
    )

    if cover_bgr is None:
        print(
            "Could not load image."
        )
        continue

    cover_rgb = cv2.cvtColor(
        cover_bgr,
        cv2.COLOR_BGR2RGB
    )

    cover_ycrcb = cv2.cvtColor(
        cover_rgb,
        cv2.COLOR_RGB2YCrCb
    )

    cover_y = cover_ycrcb[:, :, 0]

    height, width = cover_y.shape

    total_pixels = (
        height * width
    )

    blocks = divide_into_blocks(
        cover_y,
        block_size=8
    )

    predictions = predictor.predict_blocks(
        blocks
    )

    for payload_bpp in PAYLOAD_LEVELS:

        target_bits = int(
            total_pixels
            *
            payload_bpp
        )

        secret_bits = generate_payload(
            target_bits
        )

        (
            stego_y,
            embedding_map,
            bits_embedded
        ) = embed_ai(
            cover_y,
            secret_bits,
            predictions
        )

        # ----------------------------------------------------
        # Reconstruct
        # ----------------------------------------------------

        stego_ycrcb = cover_ycrcb.copy()

        stego_ycrcb[:, :, 0] = stego_y

        stego_rgb = cv2.cvtColor(
            stego_ycrcb,
            cv2.COLOR_YCrCb2RGB
        )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        extracted_bits = extract_ai(
            stego_y,
            embedding_map
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

        correct = (
            extracted_bits
            ==
            secret_bits
        )

        actual_bpp = (
            bits_embedded
            /
            total_pixels
        )

        results.append({
            "Image": image_name,
            "Target_BPP": payload_bpp,
            "Target_Bits": target_bits,
            "Bits_Embedded": bits_embedded,
            "Actual_BPP": actual_bpp,
            "MSE": mse,
            "PSNR_dB": psnr,
            "SSIM": ssim_value,
            "BER": ber,
            "Extraction_Correct": correct
        })

        print(
            f"  {payload_bpp:.2f} bpp | "
            f"embedded={bits_embedded} | "
            f"PSNR={psnr:.2f} | "
            f"SSIM={ssim_value:.4f} | "
            f"BER={ber:.6f}"
        )


# ============================================================
# Save CSV
# ============================================================

df = pd.DataFrame(
    results
)

output_path = os.path.join(
    RESULT_DIR,
    "ai_payload_experiment.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
)

print(output_path)

print("\nAverage results:")

print(
    df.groupby(
        "Target_BPP"
    )[
        [
            "Actual_BPP",
            "MSE",
            "PSNR_dB",
            "SSIM",
            "BER"
        ]
    ].mean().round(6)
)

print("=" * 70)