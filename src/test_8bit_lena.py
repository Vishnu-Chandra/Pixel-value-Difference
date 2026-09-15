import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "src")
)

from extended_epvd import (
    embed_pair_8bit,
    extract_pair_8bit
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COVER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover",
    "lena.png"
)

SECRET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "secret",
    "secret.txt"
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


def bits_from_bytes(data):

    return "".join(
        format(byte, "08b")
        for byte in data
    )


def main():

    print("=" * 75)
    print("8-BIT EXTENDED EPVD - LENA IMAGE TEST")
    print("=" * 75)

    cover = cv2.imread(
        COVER_PATH,
        cv2.IMREAD_COLOR
    )

    if cover is None:
        raise FileNotFoundError(
            f"Cover image not found: {COVER_PATH}"
        )

    with open(
        SECRET_PATH,
        "rb"
    ) as f:

        secret_data = f.read()

    secret_bits = bits_from_bytes(
        secret_data
    )

    height, width, _ = cover.shape

    print()
    print(f"Image size              : {width} x {height}")
    print(f"Secret size             : {len(secret_data)} bytes")
    print(f"Secret bits             : {len(secret_bits)}")

    # Use only complete pixel pairs.
    total_pairs = (
        height * width // 2
    )

    print(f"Available pixel pairs   : {total_pairs}")

    # Test several target BPP values.
    target_bpps = [
        1.00,
        1.25,
        1.50,
        1.75,
        2.00
    ]

    print()
    print("-" * 75)
    print(
        f"{'Target BPP':<12}"
        f"{'Embedded':<14}"
        f"{'Actual BPP':<14}"
        f"{'MSE':<14}"
        f"{'PSNR':<14}"
        f"{'SSIM':<14}"
        f"{'BER':<14}"
    )
    print("-" * 75)

    for target_bpp in target_bpps:

        # Number of bits to embed.
        target_bits = int(
            height * width * target_bpp
        )

        target_bits = min(
            target_bits,
            len(secret_bits)
        )

        bits = secret_bits[:target_bits]

        stego = cover.copy()

        extracted_bits = ""

        bit_position = 0
        successful_pairs = 0
        bit_errors = 0

        # Process pairs sequentially.
        for row in range(height):

            for col in range(0, width - 1, 2):

                if bit_position >= len(bits):
                    break

                p1 = int(
                    stego[row, col, 0]
                )

                p2 = int(
                    stego[row, col + 1, 0]
                )

                # 8-bit pair embedding.
                if (
                    len(bits) - bit_position
                    < 8
                ):
                    break

                payload = bits[
                    bit_position:
                    bit_position + 8
                ]

                new_p1, new_p2, embedded, success = (
                    embed_pair_8bit(
                        p1,
                        p2,
                        payload
                    )
                )

                if not success:
                    continue

                stego[row, col, 0] = new_p1
                stego[row, col + 1, 0] = new_p2

                extracted, count = extract_pair_8bit(
                    new_p1,
                    new_p2
                )

                extracted_bits += extracted

                bit_position += embedded
                successful_pairs += 1

            if bit_position >= len(bits):
                break

        embedded = bit_position

        if embedded > 0:

            compare_bits = bits[:embedded]

            extracted_bits = extracted_bits[:embedded]

            for a, b in zip(
                compare_bits,
                extracted_bits
            ):

                if a != b:
                    bit_errors += 1

            ber = (
                bit_errors / embedded
            )

        else:

            ber = 1.0

        mse, psnr, ssim_value = calculate_metrics(
            cover,
            stego
        )

        actual_bpp = (
            embedded /
            (height * width)
        )

        print(
            f"{target_bpp:<12.2f}"
            f"{embedded:<14}"
            f"{actual_bpp:<14.4f}"
            f"{mse:<14.4f}"
            f"{psnr:<14.4f}"
            f"{ssim_value:<14.6f}"
            f"{ber:<14.8f}"
        )

    print("-" * 75)

    print()
    print("TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()