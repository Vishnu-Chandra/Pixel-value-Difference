import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from preprocessing import (
    load_image,
    rgb_to_ycbcr,
    divide_into_blocks
)

from chaotic import generate_block_order
from epvd import extract_pair

from blind_protocol import (
    bits_to_text,
    read_header_from_block,
    read_allocation_tag
)


# =========================================================
# EXTRACT PAYLOAD FROM ONE BLOCK
# Pair 0 = allocation tag
# Pairs 1..31 = EPVD payload
# =========================================================

def extract_from_block(
    block,
    requested_bits,
    remaining_bits
):

    extracted = ""

    height, width = block.shape

    pair_number = 0

    for row in range(height):

        for col in range(0, width - 1, 2):

            # Skip Pair 0 because it contains allocation tag
            if pair_number == 0:
                pair_number += 1
                continue

            if len(extracted) >= remaining_bits:
                return extracted

            p1 = block[row, col]
            p2 = block[row, col + 1]

            try:

                expected_bits = min(
                    requested_bits,
                    remaining_bits - len(extracted)
                )

                bits, actual_bits = extract_pair(
                    p1,
                    p2,
                    expected_bits
                )

            except ValueError:

                pair_number += 1
                continue

            if actual_bits <= 0:

                pair_number += 1
                continue

            needed = (
                remaining_bits -
                len(extracted)
            )

            extracted += bits[:needed]

            pair_number += 1

    return extracted


# =========================================================
# RECEIVER-ONLY EXTRACTION
# =========================================================

def extract_secure_ai_epvd(
    stego_path,
    secret_key
):

    print("----------------------------------------")
    print("AI + Chaotic EPVD Blind Extraction")
    print("----------------------------------------")

    # -----------------------------------------------------
    # Load ONLY STEGO IMAGE
    # -----------------------------------------------------

    stego = load_image(
        stego_path
    )

    print("Stego image loaded:", stego.shape)

    # -----------------------------------------------------
    # RGB -> YCbCr
    # -----------------------------------------------------

    stego_ycbcr = rgb_to_ycbcr(
        stego
    )

    stego_Y = stego_ycbcr[:, :, 0]

    # -----------------------------------------------------
    # Divide into blocks
    # -----------------------------------------------------

    blocks = divide_into_blocks(
        stego_Y,
        block_size=8
    )

    if len(blocks) < 2:
        raise ValueError(
            "Invalid stego image: insufficient blocks."
        )

    print("Total blocks:", len(blocks))

    # -----------------------------------------------------
    # READ HEADER FROM BLOCK 0
    # -----------------------------------------------------

    try:

        header = read_header_from_block(
            blocks[0]
        )

    except ValueError as e:

        raise ValueError(str(e))

    payload_bits = int(
        header["payload_bits"]
    )

    block_size = int(
        header["block_size"]
    )

    print("Protocol version:", header["version"])
    print("Block size:", block_size)
    print("Payload bits:", payload_bits)

    if payload_bits <= 0:
        raise ValueError(
            "Invalid payload length."
        )

    # -----------------------------------------------------
    # CHAOTIC ORDER
    #
    # Block 0 is excluded.
    # -----------------------------------------------------

    data_block_count = len(blocks) - 1

    block_order = generate_block_order(
        data_block_count,
        secret_key
    )

    print(
        "Chaotic traversal generated."
    )

    # -----------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------

    extracted_bits = ""

    blocks_used = 0

    for relative_index in block_order:

        if len(extracted_bits) >= payload_bits:
            break

        block_index = relative_index + 1

        block = blocks[block_index]

        # -------------------------------------------------
        # Read AI allocation tag
        # -------------------------------------------------

        try:

            requested_bits = read_allocation_tag(
                block
            )

        except ValueError:

            continue

        remaining = (
            payload_bits -
            len(extracted_bits)
        )

        block_bits = extract_from_block(
            block,
            requested_bits,
            remaining
        )

        if len(block_bits) > 0:

            blocks_used += 1

            extracted_bits += block_bits

    # -----------------------------------------------------
    # Exact payload length
    # -----------------------------------------------------

    extracted_bits = extracted_bits[
        :payload_bits
    ]

    # -----------------------------------------------------
    # Verify complete extraction
    # -----------------------------------------------------

    complete = (
        len(extracted_bits)
        == payload_bits
    )

    if not complete:

        raise ValueError(
            f"Extraction incomplete: "
            f"{len(extracted_bits)}/{payload_bits} bits recovered."
        )

    # -----------------------------------------------------
    # Bits -> UTF-8 message
    # -----------------------------------------------------

    extracted_message = bits_to_text(
        extracted_bits
    )

    print()
    print("----------------------------------------")
    print("Blind extraction completed")
    print("----------------------------------------")

    print("Blocks used:", blocks_used)
    print("Bits extracted:", len(extracted_bits))
    print("Message:", ascii(extracted_message))

    return (
        extracted_bits,
        extracted_message,
        blocks_used
    )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    STEGO_PATH = os.path.join(
        PROJECT_ROOT,
        "results",
        "stego_blind_test.png"
    )

    result = extract_secure_ai_epvd(
        STEGO_PATH,
        "AI_EPVD_2026"
    )

    print(result)