import os
import cv2

from preprocessing import rgb_to_ycbcr
from ai_adaptive import AdaptiveBitPredictor
from epvd import extract_pair


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest_5to7.pkl"
)


def bits_to_text(bits):
    chars = []

    for i in range(0, len(bits), 8):

        byte = bits[i:i + 8]

        if len(byte) < 8:
            break

        chars.append(chr(int(byte, 2)))

    return ''.join(chars)


def extract_ai_epvd(
    cover_path,
    stego_path,
    expected_bits
):

    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)

    if cover is None:
        raise FileNotFoundError(
            f"Cover image not found: {cover_path}"
        )

    if stego is None:
        raise FileNotFoundError(
            f"Stego image not found: {stego_path}"
        )

    cover_rgb = cv2.cvtColor(
        cover,
        cv2.COLOR_BGR2RGB
    )

    stego_rgb = cv2.cvtColor(
        stego,
        cv2.COLOR_BGR2RGB
    )

    cover_ycbcr = rgb_to_ycbcr(
        cover_rgb
    )

    stego_ycbcr = rgb_to_ycbcr(
        stego_rgb
    )

    cover_y = cover_ycbcr[:, :, 0]
    stego_y = stego_ycbcr[:, :, 0]

    predictor = AdaptiveBitPredictor(
        MODEL_PATH
    )

    extracted_bits = ""
    pairs_used = 0
    blocks_used = 0

    height, width = stego_y.shape

    for i in range(0, height, 8):

        for j in range(0, width, 8):

            if len(extracted_bits) >= expected_bits:
                break

            # IMPORTANT:
            # RF allocation is regenerated from
            # the original cover block.
            cover_block = cover_y[
                i:i + 8,
                j:j + 8
            ]

            stego_block = stego_y[
                i:i + 8,
                j:j + 8
            ]

            requested_bits = predictor.predict_block(
                cover_block
            )

            block_extracted = False

            for row in range(8):

                for col in range(0, 8, 2):

                    if len(extracted_bits) >= expected_bits:
                        break

                    p1 = int(
                        stego_block[row, col]
                    )

                    p2 = int(
                        stego_block[row, col + 1]
                    )

                    remaining = (
                        expected_bits
                        - len(extracted_bits)
                    )

                    bits_to_extract = min(
                        requested_bits,
                        remaining
                    )

                    bits, actual_bits = extract_pair(
                        p1,
                        p2,
                        expected_bits=bits_to_extract
                    )

                    extracted_bits += bits

                    pairs_used += 1
                    block_extracted = True

            if block_extracted:
                blocks_used += 1

        if len(extracted_bits) >= expected_bits:
            break

    extracted_bits = extracted_bits[
        :expected_bits
    ]

    extracted_message = bits_to_text(
        extracted_bits
    )

    return (
        extracted_bits,
        extracted_message,
        pairs_used,
        blocks_used
    )


if __name__ == "__main__":

    cover_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "cover",
        "lena.png"
    )

    stego_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "stego_ai_epvd.png"
    )

    secret_message = "Hello AI EPVD"

    expected_bits = len(
        secret_message
    ) * 8

    print("-" * 50)
    print("AI-Adaptive EPVD Extraction")
    print("-" * 50)

    bits, message, pairs, blocks = \
        extract_ai_epvd(
            cover_path,
            stego_path,
            expected_bits
        )

    print(
        f"Bits extracted: {len(bits)}"
    )

    print(
        f"Expected bits: {expected_bits}"
    )

    print(
        f"Blocks used: {blocks}"
    )

    print(
        f"Pairs used: {pairs}"
    )

    print(
        f"Extracted message: {message}"
    )

    print(
        f"Original message: {secret_message}"
    )

    print(
        f"Message matches: "
        f"{message == secret_message}"
    )

    if message == secret_message:
        print(
            "\nAI-Adaptive extraction successful!"
        )
    else:
        print(
            "\nAI-Adaptive extraction failed."
        )