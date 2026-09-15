import os
import cv2

from preprocessing import divide_into_blocks
from ai_adaptive import AdaptiveBitPredictor
from epvd import extract_pair


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover",
    "lena.png"
)

STEGO_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "stego_ai_5to7.png"
)

EXPECTED_MESSAGE = "Hello AI 5to7"


def bits_to_text(bits):

    chars = []

    for i in range(
        0,
        len(bits),
        8
    ):

        byte = bits[
            i:i + 8
        ]

        if len(byte) < 8:
            break

        chars.append(
            chr(
                int(byte, 2)
            )
        )

    return "".join(chars)


# ============================================================
# Extraction
# ============================================================

def extract_ai_epvd(
    cover_path,
    stego_path,
    expected_message
):

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    cover = cv2.imread(
        cover_path
    )

    stego = cv2.imread(
        stego_path
    )

    if cover is None:
        raise FileNotFoundError(
            cover_path
        )

    if stego is None:
        raise FileNotFoundError(
            stego_path
        )

    # --------------------------------------------------------
    # Convert to YCrCb
    # --------------------------------------------------------

    cover_rgb = cv2.cvtColor(
        cover,
        cv2.COLOR_BGR2RGB
    )

    stego_rgb = cv2.cvtColor(
        stego,
        cv2.COLOR_BGR2RGB
    )

    cover_ycrcb = cv2.cvtColor(
        cover_rgb,
        cv2.COLOR_RGB2YCrCb
    )

    stego_ycrcb = cv2.cvtColor(
        stego_rgb,
        cv2.COLOR_RGB2YCrCb
    )

    cover_y = cover_ycrcb[:, :, 0]

    stego_y = stego_ycrcb[:, :, 0]

    height, width = cover_y.shape

    # --------------------------------------------------------
    # Generate AI allocation map from COVER
    # --------------------------------------------------------

    predictor = AdaptiveBitPredictor()

    cover_blocks = divide_into_blocks(
        cover_y,
        block_size=8
    )

    predictions = predictor.predict_blocks(
        cover_blocks
    )

    # --------------------------------------------------------
    # Expected payload
    # --------------------------------------------------------

    expected_bits = "".join(
        format(
            ord(char),
            "08b"
        )
        for char in expected_message
    )

    required_length = len(
        expected_bits
    )

    extracted_bits = ""

    block_index = 0
    pairs_used = 0

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

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

            if len(extracted_bits) >= required_length:
                break

            block = stego_y[
                block_row:block_row + 8,
                block_col:block_col + 8
            ]

            requested_bits = int(
                predictions[block_index]
            )

            for row in range(8):

                for col in range(
                    0,
                    8,
                    2
                ):

                    if len(extracted_bits) >= required_length:
                        break

                    p1 = int(
                        block[row, col]
                    )

                    p2 = int(
                        block[row, col + 1]
                    )

                    remaining = (
                        required_length
                        -
                        len(extracted_bits)
                    )

                    bits_to_extract = min(
                        requested_bits,
                        remaining
                    )

                    bits, actual_bits = extract_pair(
                        p1,
                        p2,
                        bits_to_extract
                    )

                    if actual_bits > 0:

                        extracted_bits += bits[
                            :bits_to_extract
                        ]

                        pairs_used += 1

            block_index += 1

        if len(extracted_bits) >= required_length:
            break

    # --------------------------------------------------------
    # Convert to text
    # --------------------------------------------------------

    extracted_message = bits_to_text(
        extracted_bits
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    match = (
        extracted_message
        ==
        expected_message
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("=" * 70)
    print("AI-ADAPTIVE EPVD EXTRACTION — 5 TO 7 BITS")
    print("=" * 70)

    print(
        f"\nExpected message : "
        f"{expected_message}"
    )

    print(
        f"Expected bits    : "
        f"{required_length}"
    )

    print(
        f"Extracted bits   : "
        f"{len(extracted_bits)}"
    )

    print(
        f"Extracted message: "
        f"{extracted_message}"
    )

    print(
        f"Pairs used       : "
        f"{pairs_used}"
    )

    print(
        f"Message matches  : "
        f"{match}"
    )

    if match:
        print(
            "\nAI-ADAPTIVE EXTRACTION SUCCESSFUL!"
        )
    else:
        print(
            "\nAI-ADAPTIVE EXTRACTION FAILED!"
        )

    print("=" * 70)


if __name__ == "__main__":

    extract_ai_epvd(
        COVER_PATH,
        STEGO_PATH,
        EXPECTED_MESSAGE
    )