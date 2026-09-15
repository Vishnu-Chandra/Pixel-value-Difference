import os
import numpy as np
import cv2
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from epvd import (
    pixel_difference,
    get_bit_capacity,
    embed_pair_adaptive,
    extract_pair
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


IMAGES = [
    "lena.png",
    "baboon.jpeg",
    "car-house.jpeg",
    "jet.jpeg",
    "peppers.jpeg",
    "sailboat.jpeg"
]


TARGET_BPP = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]


# ============================================================
# Image Utilities
# ============================================================

def load_rgb(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def rgb_to_ycrcb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2YCrCb
    )


def ycrcb_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_YCrCb2RGB
    )


# ============================================================
# Payload Generation
# ============================================================

def generate_payload(
    number_of_bits,
    seed
):

    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        2,
        size=number_of_bits,
        dtype=np.uint8
    ).tolist()


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    original,
    stego,
    original_bits,
    extracted_bits
):

    original_float = original.astype(
        np.float64
    )

    stego_float = stego.astype(
        np.float64
    )

    mse = float(
        np.mean(
            (original_float - stego_float) ** 2
        )
    )

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = float(
            10 *
            np.log10(
                (255.0 ** 2) / mse
            )
        )

    ssim_value = float(
        ssim(
            original,
            stego,
            channel_axis=2,
            data_range=255
        )
    )

    compare_length = min(
        len(original_bits),
        len(extracted_bits)
    )

    if compare_length == 0:

        ber = 1.0

    else:

        original_array = np.asarray(
            original_bits[:compare_length],
            dtype=np.uint8
        )

        extracted_array = np.asarray(
            extracted_bits[:compare_length],
            dtype=np.uint8
        )

        errors = np.sum(
            original_array != extracted_array
        )

        ber = float(
            errors / compare_length
        )

    return (
        mse,
        psnr,
        ssim_value,
        ber
    )


# ============================================================
# STANDARD EPVD EMBEDDING
# ============================================================

def embed_standard(
    cover_rgb,
    payload_bits
):

    ycrcb = rgb_to_ycrcb(
        cover_rgb
    )

    stego_ycrcb = ycrcb.copy()

    y_channel = stego_ycrcb[:, :, 0]

    height, width = y_channel.shape

    payload_position = 0

    successful_pairs = 0

    failed_pairs = 0

    embedding_map = []


    # --------------------------------------------------------
    # Sequential pair processing
    #
    # NO Random Forest
    # NO texture classification
    # NO adaptive AI decision
    #
    # Native EPVD capacity is used directly.
    # --------------------------------------------------------

    for row in range(height):

        for col in range(
            0,
            width - 1,
            2
        ):

            if payload_position >= len(
                payload_bits
            ):
                break


            p1 = int(
                y_channel[row, col]
            )

            p2 = int(
                y_channel[row, col + 1]
            )


            difference = pixel_difference(
                p1,
                p2
            )


            native_bits = get_bit_capacity(
                difference
            )


            if native_bits <= 0:

                failed_pairs += 1

                continue


            remaining_bits = (
                len(payload_bits)
                -
                payload_position
            )


            bits_to_embed = min(
                native_bits,
                remaining_bits
            )


            secret_bits = payload_bits[
                payload_position:
                payload_position
                +
                bits_to_embed
            ]


            # Convert list of bits to string
            secret_string = "".join(
                str(int(bit))
                for bit in secret_bits
            )


            try:

                new_p1, new_p2, actual_bits, success = (
                    embed_pair_adaptive(
                        p1,
                        p2,
                        secret_string,
                        bits_to_embed
                    )
                )

            except Exception:

                failed_pairs += 1

                continue


            if (
                not success
                or
                actual_bits <= 0
            ):

                failed_pairs += 1

                continue


            actual_bits = min(
                actual_bits,
                bits_to_embed
            )


            y_channel[
                row,
                col
            ] = np.uint8(
                np.clip(
                    new_p1,
                    0,
                    255
                )
            )


            y_channel[
                row,
                col + 1
            ] = np.uint8(
                np.clip(
                    new_p2,
                    0,
                    255
                )
            )


            embedding_map.append(
                (
                    row,
                    col,
                    actual_bits
                )
            )


            payload_position += actual_bits

            successful_pairs += 1


        if payload_position >= len(
            payload_bits
        ):

            break


    stego_rgb = ycrcb_to_rgb(
        stego_ycrcb
    )


    return (
        stego_rgb,
        embedding_map,
        payload_position,
        successful_pairs,
        failed_pairs
    )


# ============================================================
# STANDARD EPVD EXTRACTION
# ============================================================

def extract_standard(
    stego_rgb,
    embedding_map
):

    ycrcb = rgb_to_ycrcb(
        stego_rgb
    )

    y_channel = ycrcb[:, :, 0]

    extracted_bits = []


    for row, col, expected_bits in embedding_map:

        p1 = int(
            y_channel[row, col]
        )

        p2 = int(
            y_channel[row, col + 1]
        )


        try:

            bits, actual_bits = extract_pair(
                p1,
                p2,
                expected_bits
            )

            extracted_bits.extend(
                int(x) for x in bits[:expected_bits]
            )

        except Exception:

            continue


    return extracted_bits


# ============================================================
# Single Experiment
# ============================================================

def run_experiment(
    image_name,
    target_bpp
):

    cover_path = os.path.join(
        COVER_DIR,
        image_name
    )


    cover = load_rgb(
        cover_path
    )


    height, width = cover.shape[:2]


    target_bits = int(
        round(
            height *
            width *
            target_bpp
        )
    )


    payload = generate_payload(
        target_bits,
        seed=2026 + int(
            target_bpp * 1000
        )
    )


    (
        stego,
        embedding_map,
        embedded_bits,
        successful_pairs,
        failed_pairs
    ) = embed_standard(
        cover,
        payload
    )


    extracted_bits = extract_standard(
        stego,
        embedding_map
    )


    compare_length = min(
        len(payload),
        len(extracted_bits)
    )


    extraction_correct = (
        len(extracted_bits)
        == len(payload)
        and
        extracted_bits
        == payload
    )


    mse, psnr, ssim_value, ber = (
        calculate_metrics(
            cover,
            stego,
            payload,
            extracted_bits
        )
    )


    actual_bpp = (
        embedded_bits /
        (height * width)
    )


    output_name = (
        "standard_"
        +
        os.path.splitext(
            image_name
        )[0]
        +
        "_"
        +
        f"{target_bpp:.2f}bpp.png"
    )


    output_path = os.path.join(
        RESULT_DIR,
        output_name
    )


    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            stego,
            cv2.COLOR_RGB2BGR
        )
    )


    return {

        "Image":
            image_name,

        "Target_BPP":
            target_bpp,

        "Actual_BPP":
            actual_bpp,

        "Embedded_Bits":
            embedded_bits,

        "Target_Bits":
            target_bits,

        "MSE":
            mse,

        "PSNR_dB":
            psnr,

        "SSIM":
            ssim_value,

        "BER":
            ber,

        "Extraction_Correct":
            extraction_correct,

        "Successful_Pairs":
            successful_pairs,

        "Failed_Pairs":
            failed_pairs
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STANDARD EPVD PAYLOAD EXPERIMENT"
    )

    print("=" * 70)


    results = []


    for image_name in IMAGES:

        print(
            f"\nProcessing: {image_name}"
        )


        for target_bpp in TARGET_BPP:

            try:

                result = run_experiment(
                    image_name,
                    target_bpp
                )


                results.append(
                    result
                )


                print(
                    f"  "
                    f"{target_bpp:.2f} bpp | "
                    f"embedded="
                    f"{result['Embedded_Bits']:<7d} | "
                    f"PSNR="
                    f"{result['PSNR_dB']:.2f} | "
                    f"SSIM="
                    f"{result['SSIM']:.4f} | "
                    f"BER="
                    f"{result['BER']:.6f}"
                )


            except Exception as error:

                print(
                    f"  "
                    f"{target_bpp:.2f} bpp | "
                    f"ERROR: {error}"
                )


    if not results:

        print(
            "\nNo results generated."
        )

        return


    dataframe = pd.DataFrame(
        results
    )


    output_csv = os.path.join(
        RESULT_DIR,
        "standard_payload_experiment.csv"
    )


    dataframe.to_csv(
        output_csv,
        index=False
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        "\nResults saved to:"
    )

    print(
        output_csv
    )


    print(
        "\nAverage results:"
    )


    average = dataframe.groupby(
        "Target_BPP"
    )[
        [
            "Actual_BPP",
            "MSE",
            "PSNR_dB",
            "SSIM",
            "BER"
        ]
    ].mean()


    print(
        average.to_string()
    )


    print(
        "\nExtraction correctness:"
    )


    correctness = dataframe.groupby(
        "Target_BPP"
    )[
        "Extraction_Correct"
    ].mean()


    print(
        correctness.to_string()
    )


    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()