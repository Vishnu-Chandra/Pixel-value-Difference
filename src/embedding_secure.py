import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    load_image,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    divide_into_blocks
)

from ai_adaptive import AdaptiveBitPredictor
from chaotic import generate_block_order
from epvd import embed_pair_adaptive

from blind_protocol import (
    text_to_bits,
    write_header_to_block,
    write_allocation_tag
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# =========================================================
# EMBED PAYLOAD INTO ONE BLOCK
# Pair 0 is reserved for the allocation tag.
# Pairs 1..31 are used for EPVD payload.
# =========================================================

def embed_block(
    block,
    secret_bits,
    bit_position,
    requested_bits
):

    block = block.copy()

    height, width = block.shape

    embedded = 0

    # -----------------------------------------------------
    # Payload pairs start from pair 1.
    # Pair 0 is reserved for the 2-bit allocation tag.
    # -----------------------------------------------------

    pair_number = 0

    for row in range(height):

        for col in range(0, width - 1, 2):

            # Skip Pair 0
            if pair_number == 0:
                pair_number += 1
                continue

            if bit_position >= len(secret_bits):
                return block, bit_position, embedded

            remaining = (
                len(secret_bits) - bit_position
            )

            # Need a complete EPVD segment
            if remaining < requested_bits:
                return block, bit_position, embedded

            secret_segment = "".join(
                str(b) for b in secret_bits[
                    bit_position:
                    bit_position + requested_bits
                ]
            )

            p1 = block[row, col]
            p2 = block[row, col + 1]

            try:

                (
                    new_p1,
                    new_p2,
                    actual_bits,
                    success
                ) = embed_pair_adaptive(
                    p1,
                    p2,
                    secret_segment,
                    requested_bits
                )

            except ValueError:
                pair_number += 1
                continue

            if success:

                block[row, col] = new_p1
                block[row, col + 1] = new_p2

                bit_position += actual_bits
                embedded += actual_bits

            pair_number += 1

    return block, bit_position, embedded


# =========================================================
# MAIN RECEIVER-INDEPENDENT EMBEDDING
# =========================================================

def embed_secure_ai_epvd(
    image_path,
    output_path,
    message,
    secret_key
):

    print("----------------------------------------")
    print("AI + Chaotic EPVD Blind Embedding")
    print("----------------------------------------")

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = load_image(image_path)

    print("Original image shape:", image.shape)

    # -----------------------------------------------------
    # RGB -> YCbCr
    # -----------------------------------------------------

    ycbcr = rgb_to_ycbcr(image)

    Y = ycbcr[:, :, 0]

    # -----------------------------------------------------
    # Secret -> bits
    # -----------------------------------------------------

    secret_bits = text_to_bits(message)

    payload_length = len(secret_bits)

    print("Secret message:", message)
    print("Payload bits:", payload_length)
    print("Secret key:", secret_key)

    # -----------------------------------------------------
    # Divide into 8x8 blocks
    # -----------------------------------------------------

    blocks = divide_into_blocks(
        Y,
        block_size=8
    )

    print("Total blocks:", len(blocks))

    if len(blocks) < 2:
        raise ValueError(
            "Image is too small. At least two 8x8 blocks are required."
        )

    # -----------------------------------------------------
    # Block 0 = synchronization header
    # -----------------------------------------------------

    processed_blocks = [
        block.copy()
        for block in blocks
    ]

    processed_blocks[0] = write_header_to_block(
        processed_blocks[0],
        payload_length
    )

    print("64-bit synchronization header embedded.")

    # -----------------------------------------------------
    # AI prediction
    #
    # Important:
    # Predictions are generated from ORIGINAL cover blocks.
    # -----------------------------------------------------

    predictor = AdaptiveBitPredictor()

    predictions = predictor.predict_blocks(
        blocks
    )

    unique, counts = np.unique(
        predictions,
        return_counts=True
    )

    print("AI allocation distribution:")

    for bits, count in zip(unique, counts):

        percentage = (
            count / len(predictions) * 100
        )

        print(
            f"  {bits} bits: "
            f"{count} blocks "
            f"({percentage:.2f}%)"
        )

    # -----------------------------------------------------
    # Exclude Block 0 from chaotic payload pool
    # -----------------------------------------------------

    data_block_count = len(blocks) - 1

    block_order = generate_block_order(
        data_block_count,
        secret_key
    )

    print("Chaotic block selection enabled.")

    # -----------------------------------------------------
    # Embed message
    # -----------------------------------------------------

    bit_position = 0
    total_embedded = 0
    blocks_used = 0

    for relative_index in block_order:

        if bit_position >= payload_length:
            break

        # Convert chaotic data-block index to actual block index.
        block_index = relative_index + 1

        requested_bits = int(
            predictions[block_index]
        )

        # -------------------------------------------------
        # First reserve allocation tag
        # -------------------------------------------------

        processed_block = processed_blocks[
            block_index
        ].copy()

        processed_block = write_allocation_tag(
            processed_block,
            requested_bits
        )

        # -------------------------------------------------
        # Embed payload into pairs 1..31
        # -------------------------------------------------

        (
            processed_block,
            bit_position,
            embedded
        ) = embed_block(
            processed_block,
            secret_bits,
            bit_position,
            requested_bits
        )

        processed_blocks[
            block_index
        ] = processed_block

        total_embedded += embedded

        if embedded > 0:
            blocks_used += 1

    # -----------------------------------------------------
    # Verify
    # -----------------------------------------------------

    complete = (
        total_embedded >= payload_length
    )

    if not complete:

        raise ValueError(
            f"Message could not be completely embedded. "
            f"Embedded {total_embedded}/{payload_length} bits."
        )

    # -----------------------------------------------------
    # Reconstruct Y
    # -----------------------------------------------------

    height, width = Y.shape

    reconstructed_Y = np.zeros_like(Y)

    index = 0

    for i in range(
        0,
        height - height % 8,
        8
    ):

        for j in range(
            0,
            width - width % 8,
            8
        ):

            reconstructed_Y[
                i:i + 8,
                j:j + 8
            ] = processed_blocks[index]

            index += 1

    # -----------------------------------------------------
    # Replace Y channel
    # -----------------------------------------------------

    stego_ycbcr = ycbcr.copy()

    stego_ycbcr[
        :reconstructed_Y.shape[0],
        :reconstructed_Y.shape[1],
        0
    ] = reconstructed_Y

    # -----------------------------------------------------
    # YCbCr -> RGB
    # -----------------------------------------------------

    stego_rgb = ycbcr_to_rgb(
        stego_ycbcr
    )

    # -----------------------------------------------------
    # Save PNG
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    import cv2

    success = cv2.imwrite(
        output_path,
        cv2.cvtColor(
            stego_rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:
        raise IOError(
            "Failed to save stego image."
        )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("----------------------------------------")
    print("Blind AI + Chaotic embedding completed")
    print("----------------------------------------")

    print("Blocks used:", blocks_used)
    print("Payload bits:", payload_length)
    print("Bits embedded:", total_embedded)
    print("Complete:", complete)
    print("Stego:", output_path)

    return {
        "embedded_bits": int(total_embedded),
        "blocks_used": int(blocks_used),
        "message_bits": int(payload_length),
        "complete": bool(complete)
    }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    IMAGE_PATH = os.path.join(
        PROJECT_ROOT,
        "data",
        "cover",
        "lena.png"
    )

    OUTPUT_PATH = os.path.join(
        PROJECT_ROOT,
        "results",
        "stego_blind_test.png"
    )

    result = embed_secure_ai_epvd(
        IMAGE_PATH,
        OUTPUT_PATH,
        "Hello Secure AI EPVD",
        "AI_EPVD_2026"
    )

    print(result)