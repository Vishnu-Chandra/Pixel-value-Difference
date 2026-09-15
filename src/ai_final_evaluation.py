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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COVER_DIR = os.path.join(PROJECT_ROOT, "data", "cover")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


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


def generate_secret_bits(length, seed=2026):

    rng = np.random.default_rng(seed)

    return "".join(
        rng.choice(["0", "1"], size=length)
    )


def calculate_metrics(original, stego):

    original_float = original.astype(np.float64)
    stego_float = stego.astype(np.float64)

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


def calculate_ber(original_bits, extracted_bits):

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

    return errors / compare_length, errors


def embed_ai(ycrcb, secret_bits, predictor):

    stego = ycrcb.copy()

    height, width, _ = stego.shape

    bit_pointer = 0

    successful_pairs = 0
    failed_pairs = 0

    embedding_map = []

    # AI version uses Y channel for adaptive embedding
    channel = 0

    channel_data = stego[:, :, channel]

    # Create 8x8 blocks
    block_size = 8

    blocks = []
    block_locations = []

    for row in range(0, height - block_size + 1, block_size):

        for col in range(
            0,
            width - block_size + 1,
            block_size
        ):

            block = channel_data[
                row:row + block_size,
                col:col + block_size
            ]

            blocks.append(block.copy())
            block_locations.append((row, col))

    # RF predicts 5/6/7 bits for each block
    predictions = predictor.predict_blocks(blocks)

    for block_index, requested_bits in enumerate(predictions):

        if bit_pointer >= len(secret_bits):
            break

        block_row, block_col = block_locations[block_index]

        block = channel_data[
            block_row:block_row + block_size,
            block_col:block_col + block_size
        ]

        for local_row in range(block_size):

            if bit_pointer >= len(secret_bits):
                break

            for local_col in range(0, block_size - 1, 2):

                if bit_pointer >= len(secret_bits):
                    break

                row = block_row + local_row
                col = block_col + local_col

                p1 = int(channel_data[row, col])
                p2 = int(channel_data[row, col + 1])

                new_p1, new_p2, embedded, success = (
                    embed_pair_adaptive(
                        p1,
                        p2,
                        secret_bits[bit_pointer:],
                        int(requested_bits)
                    )
                )

                if success and embedded > 0:

                    channel_data[row, col] = new_p1
                    channel_data[row, col + 1] = new_p2

                    embedding_map.append(
                        (
                            channel,
                            row,
                            col,
                            embedded
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
        embedding_map
    )


def extract_using_map(
    ycrcb,
    embedding_map,
    total_bits
):

    extracted = []
    extracted_count = 0

    for channel, row, col, expected_bits in embedding_map:

        if extracted_count >= total_bits:
            break

        p1 = int(ycrcb[row, col, channel])
        p2 = int(ycrcb[row, col + 1, channel])

        bits, actual_bits = extract_pair(
            p1,
            p2,
            expected_bits
        )

        remaining = total_bits - extracted_count

        use_bits = min(
            actual_bits,
            remaining
        )

        extracted.append(
            bits[:use_bits]
        )

        extracted_count += use_bits

    return "".join(extracted)


def process_image(
    image_name,
    predictor
):

    print("\n" + "=" * 70)
    print(f"IMAGE: {image_name}")
    print("=" * 70)

    cover_path = os.path.join(
        COVER_DIR,
        image_name
    )

    cover_rgb = load_image(
        cover_path
    )

    height, width, _ = cover_rgb.shape

    total_pixels = height * width

    ycrcb_cover = rgb_to_ycbcr(
        cover_rgb
    )

    output_csv = os.path.join(
        RESULTS_DIR,
        f"ai_final_{os.path.splitext(image_name)[0]}.csv"
    )

    rows = []

    for bpp in BPP_LEVELS:

        print("\n" + "-" * 70)
        print(f"Testing {bpp:.2f} BPP")
        print("-" * 70)

        target_bits = int(
            total_pixels * bpp
        )

        secret_bits = generate_secret_bits(
            target_bits,
            seed=2026 + int(bpp * 100)
        )

        (
            stego_ycrcb,
            embedded_bits,
            successful_pairs,
            failed_pairs,
            embedding_map
        ) = embed_ai(
            ycrcb_cover,
            secret_bits,
            predictor
        )

        actual_bpp = (
            embedded_bits / total_pixels
        )

        # --------------------------------------------------
        # DIRECT YCrCb EXTRACTION
        # --------------------------------------------------

        direct_extracted = extract_using_map(
            stego_ycrcb,
            embedding_map,
            embedded_bits
        )

        direct_ber, direct_errors = calculate_ber(
            secret_bits[:embedded_bits],
            direct_extracted
        )

        direct_exact = (
            secret_bits[:embedded_bits]
            == direct_extracted
        )

        # --------------------------------------------------
        # RGB STEGO IMAGE
        # --------------------------------------------------

        stego_rgb = ycbcr_to_rgb(
            stego_ycrcb
        )

        mse, psnr, ssim_value = calculate_metrics(
            cover_rgb,
            stego_rgb
        )

        # --------------------------------------------------
        # RGB -> YCrCb ROUND TRIP
        # --------------------------------------------------

        roundtrip_ycrcb = rgb_to_ycbcr(
            stego_rgb
        )

        roundtrip_extracted = extract_using_map(
            roundtrip_ycrcb,
            embedding_map,
            embedded_bits
        )

        roundtrip_ber, roundtrip_errors = calculate_ber(
            secret_bits[:embedded_bits],
            roundtrip_extracted
        )

        roundtrip_exact = (
            secret_bits[:embedded_bits]
            == roundtrip_extracted
        )

        # --------------------------------------------------
        # SAVE STEGO IMAGE
        # --------------------------------------------------

        output_name = (
            f"ai_{os.path.splitext(image_name)[0]}"
            f"_{bpp:.2f}bpp.png"
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

        # --------------------------------------------------
        # PRINT RESULTS
        # --------------------------------------------------

        print(
            f"Target bits              : {target_bits}"
        )

        print(
            f"Embedded bits            : {embedded_bits}"
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

        print("\nDIRECT YCrCb EXTRACTION")

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

        print("\nRGB STEGO QUALITY")

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

        print("\nAFTER RGB ROUND-TRIP")

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

            "Image": image_name,

            "BPP": bpp,

            "Target_Bits": target_bits,

            "Embedded_Bits": embedded_bits,

            "Actual_BPP": actual_bpp,

            "Successful_Pairs":
                successful_pairs,

            "Failed_Pairs":
                failed_pairs,

            "MSE": mse,

            "PSNR": psnr,

            "SSIM": ssim_value,

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
                roundtrip_exact
        })

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
        writer.writerows(rows)

    print(
        f"\nCSV saved: {output_csv}"
    )


def main():

    print("=" * 70)
    print("AI-EPVD FINAL EVALUATION")
    print("=" * 70)

    predictor = AdaptiveBitPredictor()

    for image_name in IMAGES:

        process_image(
            image_name,
            predictor
        )

    print("\n" + "=" * 70)
    print("AI-EPVD EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()