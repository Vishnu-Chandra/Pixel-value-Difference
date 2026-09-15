import os
import cv2
import numpy as np

from preprocessing import divide_into_blocks
from ai_adaptive import AdaptiveBitPredictor
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
    "stego_ai_5to7.png"
)

SECRET_MESSAGE = "Hello AI 5to7"


def text_to_bits(text):
    return "".join(
        format(ord(char), "08b")
        for char in text
    )


def embed_ai_epvd(
    cover_path,
    output_path,
    message
):

    image = cv2.imread(cover_path)

    if image is None:
        raise FileNotFoundError(
            cover_path
        )

    # --------------------------------------------------------
    # BGR -> RGB -> YCrCb
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    ycrcb = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2YCrCb
    )

    y_channel = ycrcb[:, :, 0].copy()

    height, width = y_channel.shape

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    blocks = divide_into_blocks(
        y_channel,
        block_size=8
    )

    # --------------------------------------------------------
    # AI predictions
    # --------------------------------------------------------

    predictor = AdaptiveBitPredictor()

    predictions = predictor.predict_blocks(
        blocks
    )

    # --------------------------------------------------------
    # Secret
    # --------------------------------------------------------

    secret_bits = text_to_bits(message)

    bit_index = 0

    successful_pairs = 0
    failed_pairs = 0
    blocks_used = 0

    # --------------------------------------------------------
    # Process every block
    # --------------------------------------------------------

    block_index = 0

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

            block = y_channel[
                block_row:block_row + 8,
                block_col:block_col + 8
            ].copy()

            requested_bits = int(
                predictions[block_index]
            )

            block_embedded = False

            # ------------------------------------------------
            # Horizontal pairs
            # ------------------------------------------------

            for row in range(8):

                for col in range(0, 8, 2):

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

                    payload_length = min(
                        requested_bits,
                        remaining
                    )

                    payload = secret_bits[
                        bit_index:
                        bit_index + payload_length
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

                        bit_index += actual_bits

                        successful_pairs += 1

                        block_embedded = True

                    else:

                        failed_pairs += 1

            # ------------------------------------------------
            # Write modified block
            # ------------------------------------------------

            y_channel[
                block_row:block_row + 8,
                block_col:block_col + 8
            ] = block

            if block_embedded:
                blocks_used += 1

            block_index += 1

        if bit_index >= len(secret_bits):
            break

    # --------------------------------------------------------
    # Reconstruct image
    # --------------------------------------------------------

    ycrcb[:, :, 0] = y_channel

    rgb_stego = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2RGB
    )

    bgr_stego = cv2.cvtColor(
        rgb_stego,
        cv2.COLOR_RGB2BGR
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        bgr_stego
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("=" * 70)
    print("AI-ADAPTIVE EPVD EMBEDDING — 5 TO 7 BITS")
    print("=" * 70)

    print(
        f"\nSecret message   : {message}"
    )

    print(
        f"Secret bits      : {len(secret_bits)}"
    )

    print(
        f"Bits embedded    : {bit_index}"
    )

    print(
        f"Remaining bits   : "
        f"{len(secret_bits) - bit_index}"
    )

    print(
        f"Successful pairs : {successful_pairs}"
    )

    print(
        f"Failed pairs     : {failed_pairs}"
    )

    print(
        f"Blocks used      : {blocks_used}"
    )

    print(
        f"\nSaved stego image:"
    )

    print(output_path)

    print("=" * 70)


if __name__ == "__main__":

    embed_ai_epvd(
        COVER_PATH,
        OUTPUT_PATH,
        SECRET_MESSAGE
    )