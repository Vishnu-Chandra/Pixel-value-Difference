import os
import cv2

from preprocessing import rgb_to_ycbcr
from epvd import extract_pair


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def bits_to_text(bits):
    """
    Convert a binary string into text.
    """

    chars = []

    for i in range(0, len(bits), 8):

        byte = bits[i:i + 8]

        if len(byte) < 8:
            break

        chars.append(
            chr(int(byte, 2))
        )

    return ''.join(chars)


def extract_standard_epvd(
    cover_path,
    stego_path,
    expected_bits
):
    """
    Extract secret data from a Standard EPVD stego image.

    The same horizontal pixel-pair traversal used during
    embedding is followed during extraction.
    """

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    cover_rgb = cv2.cvtColor(
        cover,
        cv2.COLOR_BGR2RGB
    )

    stego_rgb = cv2.cvtColor(
        stego,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Convert RGB → YCrCb
    #
    # Your preprocessing.py uses:
    #
    # cv2.COLOR_RGB2YCrCb
    #
    # Therefore channel 0 is Y.
    # --------------------------------------------------------

    cover_ycbcr = rgb_to_ycbcr(
        cover_rgb
    )

    stego_ycbcr = rgb_to_ycbcr(
        stego_rgb
    )

    cover_y = cover_ycbcr[:, :, 0]

    stego_y = stego_ycbcr[:, :, 0]

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    extracted_bits = ""

    height, width = stego_y.shape

    pairs_used = 0

    for i in range(
        0,
        height,
        8
    ):

        for j in range(
            0,
            width,
            8
        ):

            if len(extracted_bits) >= expected_bits:
                break

            stego_block = stego_y[
                i:i + 8,
                j:j + 8
            ]

            cover_block = cover_y[
                i:i + 8,
                j:j + 8
            ]

            # Same pair traversal as embedding
            for row in range(8):

                for col in range(
                    0,
                    8,
                    2
                ):

                    if len(extracted_bits) >= expected_bits:
                        break

                    p1 = int(
                        stego_block[row, col]
                    )

                    p2 = int(
                        stego_block[row, col + 1]
                    )

                    # Determine how many bits the original
                    # EPVD pair could carry.
                    #
                    # Use the COVER pair to determine the
                    # original difference range.

                    original_p1 = int(
                        cover_block[row, col]
                    )

                    original_p2 = int(
                        cover_block[row, col + 1]
                    )

                    original_difference = abs(
                        original_p2 - original_p1
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
                        - len(extracted_bits)
                    )

                    bits_to_extract = min(
                        native_bits,
                        remaining
                    )

                    bits, actual_bits = extract_pair(
                        p1,
                        p2,
                        expected_bits=bits_to_extract
                    )

                    extracted_bits += bits

                    pairs_used += 1

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
        pairs_used
    )


# ============================================================
# Test
# ============================================================

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
        "stego_standard_epvd.png"
    )

    secret_message = "Hello Standard EPVD"

    expected_bits = len(
        secret_message
    ) * 8

    print("-" * 50)
    print("Standard EPVD Extraction")
    print("-" * 50)

    bits, message, pairs = extract_standard_epvd(
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
        f"Extracted message: {message}"
    )

    print(
        f"Original message: {secret_message}"
    )

    print(
        f"Message matches: "
        f"{message == secret_message}"
    )

    print(
        f"Pairs used: {pairs}"
    )

    if message == secret_message:
        print(
            "\nStandard EPVD extraction successful!"
        )
    else:
        print(
            "\nExtraction failed."
        )