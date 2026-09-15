import os
import cv2

from preprocessing import rgb_to_ycbcr, ycbcr_to_rgb
from epvd import embed_pair_adaptive


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
    "stego_standard_epvd.png"
)

SECRET_MESSAGE = "Hello Standard EPVD"


def text_to_bits(text):
    return ''.join(
        format(ord(char), '08b')
        for char in text
    )


def embed_standard_epvd(
    y_channel,
    secret_bits
):
    stego = y_channel.copy()

    bit_index = 0
    pairs_used = 0

    height, width = stego.shape

    for i in range(0, height, 8):

        for j in range(0, width, 8):

            if bit_index >= len(secret_bits):
                break

            block = stego[
                i:i + 8,
                j:j + 8
            ].copy()

            for row in range(8):

                for col in range(0, 8, 2):

                    if bit_index >= len(secret_bits):
                        break

                    p1 = int(block[row, col])
                    p2 = int(block[row, col + 1])

                    difference = abs(p2 - p1)

                    # Native EPVD capacity
                    if difference <= 31:
                        capacity = 5
                    elif difference <= 63:
                        capacity = 5
                    elif difference <= 127:
                        capacity = 6
                    else:
                        capacity = 7

                    remaining = (
                        len(secret_bits) - bit_index
                    )

                    bits_to_use = min(
                        capacity,
                        remaining
                    )

                    payload = secret_bits[
                        bit_index:
                        bit_index + bits_to_use
                    ]

                    new_p1, new_p2, actual_bits, success = \
                        embed_pair_adaptive(
                            p1,
                            p2,
                            payload,
                            capacity
                        )

                    if success:

                        block[row, col] = new_p1
                        block[row, col + 1] = new_p2

                        bit_index += actual_bits
                        pairs_used += 1

            stego[
                i:i + 8,
                j:j + 8
            ] = block

    return stego, bit_index, pairs_used


def main():

    print("-" * 50)
    print("Standard EPVD Embedding")
    print("-" * 50)

    image = cv2.imread(COVER_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {COVER_PATH}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    ycbcr = rgb_to_ycbcr(image)

    y_channel = ycbcr[:, :, 0].copy()

    secret_bits = text_to_bits(
        SECRET_MESSAGE
    )

    print(
        f"Secret message: {SECRET_MESSAGE}"
    )

    print(
        f"Secret bits: {len(secret_bits)}"
    )

    stego_y, embedded_bits, pairs_used = \
        embed_standard_epvd(
            y_channel,
            secret_bits
        )

    ycbcr[:, :, 0] = stego_y

    stego_rgb = ycbcr_to_rgb(
        ycbcr
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

    print(
        f"Bits embedded: {embedded_bits}"
    )

    print(
        f"Pairs used: {pairs_used}"
    )

    print(
        f"Remaining bits: "
        f"{len(secret_bits) - embedded_bits}"
    )

    print(
        f"\nStego image saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()