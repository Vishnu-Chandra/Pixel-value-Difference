import os
import sys
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_image, rgb_to_ycbcr, ycbcr_to_rgb
from epvd import embed_pair_adaptive, extract_pair
from ai_adaptive import AdaptiveBitPredictor
from chaotic import generate_block_order


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

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGES = [
    "lena.png",
    "baboon.jpeg",
    "car-house.jpeg",
    "jet.jpeg",
    "peppers.jpeg",
    "sailboat.jpeg"
]

BPP_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]

SECRET_KEY = "AI_EPVD_2026"

BLOCK_SIZE = 8


# ============================================================
# SECRET GENERATION
# ============================================================

def generate_secret_bits(length, seed=2026):

    rng = np.random.default_rng(seed)

    return "".join(
        rng.choice(
            ["0", "1"],
            size=length
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(original, stego):

    original_float = original.astype(
        np.float64
    )

    stego_float = stego.astype(
        np.float64
    )

    mse = np.mean(
        (original_float - stego_float) ** 2
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(
            (255 ** 2) / mse
        )

    ssim_value = ssim(
        original,
        stego,
        channel_axis=2,
        data_range=255
    )

    return mse, psnr, ssim_value


def calculate_ber(
    original_bits,
    extracted_bits
):

    compare_length = min(
        len(original_bits),
        len(extracted_bits)
    )

    if compare_length == 0:
        return 0.0, 0

    errors = sum(
        a != b
        for a, b in zip(
            original_bits[:compare_length],
            extracted_bits[:compare_length]
        )
    )

    ber = errors / compare_length

    return ber, errors


# ============================================================
# AI + CHAOTIC EMBEDDING
# ============================================================

def embed_ai_chaotic(
    ycrcb,
    secret_bits,
    predictor,
    secret_key
):

    stego = ycrcb.copy()

    height, width, _ = stego.shape

    channel = 0

    channel_data = stego[:, :, channel]

    # --------------------------------------------------------
    # BUILD 8x8 BLOCKS
    # --------------------------------------------------------

    blocks = []
    block_locations = []

    for row in range(
        0,
        height - BLOCK_SIZE + 1,
        BLOCK_SIZE
    ):

        for col in range(
            0,
            width - BLOCK_SIZE + 1,
            BLOCK_SIZE
        ):

            block = channel_data[
                row:row + BLOCK_SIZE,
                col:col + BLOCK_SIZE
            ]

            blocks.append(
                block.copy()
            )

            block_locations.append(
                (row, col)
            )

    total_blocks = len(blocks)

    # --------------------------------------------------------
    # RANDOM FOREST PREDICTIONS
    # --------------------------------------------------------

    predictions = predictor.predict_blocks(
        blocks
    )

    # --------------------------------------------------------
    # CHAOTIC BLOCK ORDER
    # --------------------------------------------------------

    chaotic_order = generate_block_order(
        total_blocks,
        secret_key
    )

    print(
        f"Total 8x8 blocks        : {total_blocks}"
    )

    print(
        "First 10 chaotic blocks :",
        chaotic_order[:10]
    )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    bit_pointer = 0

    successful_pairs = 0
    failed_pairs = 0

    embedding_map = []

    for block_index in chaotic_order:

        if bit_pointer >= len(secret_bits):
            break

        requested_bits = int(
            predictions[block_index]
        )

        block_row, block_col = (
            block_locations[block_index]
        )

        for local_row in range(
            BLOCK_SIZE
        ):

            if bit_pointer >= len(secret_bits):
                break

            for local_col in range(
                0,
                BLOCK_SIZE - 1,
                2
            ):

                if bit_pointer >= len(secret_bits):
                    break

                row = (
                    block_row +
                    local_row
                )

                col = (
                    block_col +
                    local_col
                )

                p1 = int(
                    channel_data[
                        row,
                        col
                    ]
                )

                p2 = int(
                    channel_data[
                        row,
                        col + 1
                    ]
                )

                (
                    new_p1,
                    new_p2,
                    embedded,
                    success
                ) = embed_pair_adaptive(
                    p1,
                    p2,
                    secret_bits[
                        bit_pointer:
                    ],
                    requested_bits
                )

                if success and embedded > 0:

                    channel_data[
                        row,
                        col
                    ] = new_p1

                    channel_data[
                        row,
                        col + 1
                    ] = new_p2

                    embedding_map.append(
                        (
                            channel,
                            row,
                            col,
                            embedded,
                            block_index,
                            requested_bits
                        )
                    )

                    bit_pointer += embedded

                    successful_pairs += 1

                else:

                    failed_pairs += 1

    return (
        stego,
        bit_pointer,
        successful_pairs,
        failed_pairs,
        embedding_map,
        predictions,
        chaotic_order
    )


# ============================================================
# EXTRACTION USING EMBEDDING MAP
# ============================================================

def extract_using_map(
    ycrcb,
    embedding_map,
    total_bits
):

    extracted = []

    extracted_count = 0

    for (
        channel,
        row,
        col,
        expected_bits,
        block_index,
        requested_bits
    ) in embedding_map:

        if extracted_count >= total_bits:
            break

        p1 = int(
            ycrcb[
                row,
                col,
                channel
            ]
        )

        p2 = int(
            ycrcb[
                row,
                col + 1,
                channel
            ]
        )

        bits, actual_bits = extract_pair(
            p1,
            p2,
            expected_bits
        )

        remaining = (
            total_bits -
            extracted_count
        )

        use_bits = min(
            actual_bits,
            remaining
        )

        extracted.append(
            bits[:use_bits]
        )

        extracted_count += use_bits

    return "".join(
        extracted
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_name,
    predictor
):

    print("\n" + "=" * 70)
    print(
        f"IMAGE: {image_name}"
    )
    print("=" * 70)

    cover_path = os.path.join(
        COVER_DIR,
        image_name
    )

    cover_rgb = load_image(
        cover_path
    )

    height, width, _ = (
        cover_rgb.shape
    )

    total_pixels = (
        height * width
    )

    ycrcb_cover = rgb_to_ycbcr(
        cover_rgb
    )

    output_csv = os.path.join(
        RESULTS_DIR,
        "ai_chaotic_final_" +
        os.path.splitext(
            image_name
        )[0] +
        ".csv"
    )

    rows = []

    for bpp in BPP_LEVELS:

        print("\n" + "-" * 70)
        print(
            f"Testing {bpp:.2f} BPP"
        )
        print("-" * 70)

        target_bits = int(
            total_pixels * bpp
        )

        secret_bits = (
            generate_secret_bits(
                target_bits,
                seed=2026 +
                int(bpp * 100)
            )
        )

        (
            stego_ycrcb,
            embedded_bits,
            successful_pairs,
            failed_pairs,
            embedding_map,
            predictions,
            chaotic_order
        ) = embed_ai_chaotic(
            ycrcb_cover,
            secret_bits,
            predictor,
            SECRET_KEY
        )

        actual_bpp = (
            embedded_bits /
            total_pixels
        )

        # ----------------------------------------------------
        # DIRECT YCrCb EXTRACTION
        # ----------------------------------------------------

        direct_extracted = (
            extract_using_map(
                stego_ycrcb,
                embedding_map,
                embedded_bits
            )
        )

        direct_ber, direct_errors = (
            calculate_ber(
                secret_bits[:embedded_bits],
                direct_extracted
            )
        )

        direct_exact = (
            secret_bits[:embedded_bits]
            ==
            direct_extracted
        )

        # ----------------------------------------------------
        # RGB STEGO
        # ----------------------------------------------------

        stego_rgb = ycbcr_to_rgb(
            stego_ycrcb
        )

        mse, psnr, ssim_value = (
            calculate_metrics(
                cover_rgb,
                stego_rgb
            )
        )

        # ----------------------------------------------------
        # RGB -> YCrCb ROUND TRIP
        # ----------------------------------------------------

        roundtrip_ycrcb = (
            rgb_to_ycbcr(
                stego_rgb
            )
        )

        roundtrip_extracted = (
            extract_using_map(
                roundtrip_ycrcb,
                embedding_map,
                embedded_bits
            )
        )

        roundtrip_ber, roundtrip_errors = (
            calculate_ber(
                secret_bits[:embedded_bits],
                roundtrip_extracted
            )
        )

        roundtrip_exact = (
            secret_bits[:embedded_bits]
            ==
            roundtrip_extracted
        )

        # ----------------------------------------------------
        # BIT DISTRIBUTION
        # ----------------------------------------------------

        prediction_counts = {
            5: int(
                np.sum(predictions == 5)
            ),
            6: int(
                np.sum(predictions == 6)
            ),
            7: int(
                np.sum(predictions == 7)
            )
        }

        # ----------------------------------------------------
        # SAVE STEGO IMAGE
        # ----------------------------------------------------

        output_name = (
            "ai_chaotic_" +
            os.path.splitext(
                image_name
            )[0] +
            "_" +
            f"{bpp:.2f}bpp.png"
        )

        output_path = os.path.join(
            RESULTS_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                stego_rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Target bits              : "
            f"{target_bits}"
        )

        print(
            f"Embedded bits            : "
            f"{embedded_bits}"
        )

        print(
            f"Actual BPP               : "
            f"{actual_bpp:.6f}"
        )

        print(
            f"Successful pairs         : "
            f"{successful_pairs}"
        )

        print(
            f"Failed/skipped pairs     : "
            f"{failed_pairs}"
        )

        print(
            "\nRANDOM FOREST BIT ALLOCATION"
        )

        print(
            f"  5-bit blocks           : "
            f"{prediction_counts[5]}"
        )

        print(
            f"  6-bit blocks           : "
            f"{prediction_counts[6]}"
        )

        print(
            f"  7-bit blocks           : "
            f"{prediction_counts[7]}"
        )

        print(
            "\nDIRECT YCrCb EXTRACTION"
        )

        print(
            f"  Extracted bits         : "
            f"{len(direct_extracted)}"
        )

        print(
            f"  BER                    : "
            f"{direct_ber:.8f}"
        )

        print(
            f"  Bit errors             : "
            f"{direct_errors}"
        )

        print(
            f"  Exact extraction       : "
            f"{direct_exact}"
        )

        print(
            "\nRGB STEGO QUALITY"
        )

        print(
            f"  MSE                    : "
            f"{mse:.6f}"
        )

        print(
            f"  PSNR                   : "
            f"{psnr:.4f} dB"
        )

        print(
            f"  SSIM                   : "
            f"{ssim_value:.6f}"
        )

        print(
            "\nAFTER RGB ROUND-TRIP"
        )

        print(
            f"  Extracted bits         : "
            f"{len(roundtrip_extracted)}"
        )

        print(
            f"  BER                    : "
            f"{roundtrip_ber:.8f}"
        )

        print(
            f"  Bit errors             : "
            f"{roundtrip_errors}"
        )

        print(
            f"  Exact extraction       : "
            f"{roundtrip_exact}"
        )

        rows.append({

            "Image":
                image_name,

            "Target_BPP":
                bpp,

            "Target_Bits":
                target_bits,

            "Embedded_Bits":
                embedded_bits,

            "Actual_BPP":
                actual_bpp,

            "Successful_Pairs":
                successful_pairs,

            "Failed_Pairs":
                failed_pairs,

            "RF_5bit_Blocks":
                prediction_counts[5],

            "RF_6bit_Blocks":
                prediction_counts[6],

            "RF_7bit_Blocks":
                prediction_counts[7],

            "MSE":
                mse,

            "PSNR":
                psnr,

            "SSIM":
                ssim_value,

            "Direct_YCrCb_BER":
                direct_ber,

            "Direct_Bit_Errors":
                direct_errors,

            "Direct_Exact":
                direct_exact,

            "Roundtrip_RGB_BER":
                roundtrip_ber,

            "Roundtrip_Bit_Errors":
                roundtrip_errors,

            "Roundtrip_Exact":
                roundtrip_exact,

            "Secret_Key":
                SECRET_KEY
        })

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"\nCSV saved: {output_csv}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AI-EPVD + CHAOTIC BLOCK ORDER EVALUATION"
    )
    print("=" * 70)

    print(
        f"Secret key: {SECRET_KEY}"
    )

    predictor = (
        AdaptiveBitPredictor()
    )

    for image_name in IMAGES:

        process_image(
            image_name,
            predictor
        )

    print("\n" + "=" * 70)
    print(
        "AI-EPVD + CHAOTIC EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()