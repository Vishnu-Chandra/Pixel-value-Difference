"""
STANDARD EPVD: DIRECT YCrCb vs RGB ROUND-TRIP DIAGNOSTIC

Run from src:
    python standard_ycrcb_diagnostic.py

Purpose:
1. Embed Standard EPVD directly in YCrCb.
2. Extract immediately from the modified YCrCb array.
3. Convert the stego image YCrCb -> RGB -> YCrCb.
4. Extract again.
5. Compare BER before and after the color-space round trip.

If Direct BER = 0 but Round-trip BER > 0:
    the EPVD embedding/extraction is internally correct, and
    the errors are introduced by RGB <-> YCrCb conversion.

Tests Lena at 0.25, 0.50, 1.00 and 1.50 BPP.
"""

import os
import numpy as np
import cv2

from epvd import (
    pixel_difference,
    get_bit_capacity,
    embed_pair_adaptive,
    extract_pair
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

COVER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover",
    "lena.png"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

TEST_BPP = [0.25, 0.50, 1.00, 1.50]


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


def generate_payload(n_bits, seed):

    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        2,
        size=n_bits,
        dtype=np.uint8
    ).tolist()


def calculate_ber(original_bits, extracted_bits):
    compare_len = min(len(original_bits), len(extracted_bits))

    if compare_len == 0:
        return 0.0, 0

    errors = sum(
        a != b
        for a, b in zip(
            original_bits[:compare_len],
            extracted_bits[:compare_len]
        )
    )

    ber = errors / compare_len
    return ber, errors


def embed_standard_ycrcb(
    cover_ycrcb,
    payload_bits
):

    stego_ycrcb = cover_ycrcb.copy()

    y = stego_ycrcb[:, :, 0]

    height, width = y.shape

    position = 0

    embedding_map = []

    failed = 0

    for row in range(height):

        for col in range(0, width - 1, 2):

            if position >= len(payload_bits):
                break

            p1 = int(y[row, col])
            p2 = int(y[row, col + 1])

            original_difference = pixel_difference(
                p1,
                p2
            )

            native_bits = get_bit_capacity(
                original_difference
            )

            remaining = (
                len(payload_bits) - position
            )

            bits_to_embed = min(
                native_bits,
                remaining
            )

            chunk = payload_bits[
                position:
                position + bits_to_embed
            ]

            secret_string = "".join(
                str(int(x))
                for x in chunk
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

                failed += 1
                continue

            if not success or actual_bits <= 0:

                failed += 1
                continue

            y[row, col] = np.uint8(
                np.clip(new_p1, 0, 255)
            )

            y[row, col + 1] = np.uint8(
                np.clip(new_p2, 0, 255)
            )

            new_difference = pixel_difference(
                new_p1,
                new_p2
            )

            embedding_map.append({
                "row": row,
                "col": col,
                "bits": actual_bits,
                "native_bits": native_bits,
                "original_difference": original_difference,
                "embedded_difference": new_difference,
                "payload_start": position,
                "payload_end": (
                    position + actual_bits
                )
            })

            position += actual_bits

        if position >= len(payload_bits):
            break

    return (
        stego_ycrcb,
        embedding_map,
        position,
        failed
    )


def extract_from_ycrcb(
    stego_ycrcb,
    embedding_map
):

    y = stego_ycrcb[:, :, 0]

    extracted = []

    pair_details = []

    for item in embedding_map:

        row = item["row"]
        col = item["col"]
        expected_bits = item["bits"]

        p1 = int(y[row, col])
        p2 = int(y[row, col + 1])

        difference = pixel_difference(
            p1,
            p2
        )

        try:

            bits, actual_bits = extract_pair(
                p1,
                p2,
                expected_bits
            )

            bits = list(bits)[:expected_bits]

            extracted.extend(
                int(x) for x in bits
            )

            pair_details.append({
                **item,
                "read_p1": p1,
                "read_p2": p2,
                "read_difference": difference,
                "extracted": "".join(
                    str(x) for x in bits
                )
            })

        except Exception as error:

            pair_details.append({
                **item,
                "read_p1": p1,
                "read_p2": p2,
                "read_difference": difference,
                "extracted": "",
                "error": str(error)
            })

    return extracted, pair_details


def compare_pair_differences(
    direct_ycrcb,
    roundtrip_ycrcb,
    embedding_map
):

    direct_y = direct_ycrcb[:, :, 0]
    roundtrip_y = roundtrip_ycrcb[:, :, 0]

    changed_pixels = 0
    changed_pairs = 0
    total_pairs = len(embedding_map)

    max_y_change = 0

    range_changed_pairs = 0

    for item in embedding_map:

        row = item["row"]
        col = item["col"]

        p1_direct = int(
            direct_y[row, col]
        )

        p2_direct = int(
            direct_y[row, col + 1]
        )

        p1_round = int(
            roundtrip_y[row, col]
        )

        p2_round = int(
            roundtrip_y[row, col + 1]
        )

        if p1_direct != p1_round:
            changed_pixels += 1

        if p2_direct != p2_round:
            changed_pixels += 1

        max_y_change = max(
            max_y_change,
            abs(p1_direct - p1_round),
            abs(p2_direct - p2_round)
        )

        d1 = pixel_difference(
            p1_direct,
            p2_direct
        )

        d2 = pixel_difference(
            p1_round,
            p2_round
        )

        if d1 != d2:
            changed_pairs += 1

        try:

            c1 = get_bit_capacity(d1)
            c2 = get_bit_capacity(d2)

            if c1 != c2:
                range_changed_pairs += 1

        except Exception:
            pass

    return (
        changed_pixels,
        changed_pairs,
        total_pairs,
        max_y_change,
        range_changed_pairs
    )


def main():

    print("=" * 70)
    print(
        "STANDARD EPVD YCrCb ROUND-TRIP DIAGNOSTIC"
    )
    print("=" * 70)

    cover_rgb = load_rgb(
        COVER_PATH
    )

    cover_ycrcb = rgb_to_ycrcb(
        cover_rgb
    )

    height, width = cover_rgb.shape[:2]

    print(
        f"\nCover: lena.png"
    )

    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    for bpp in TEST_BPP:

        print("\n" + "-" * 70)
        print(
            f"Testing {bpp:.2f} BPP"
        )
        print("-" * 70)

        target_bits = int(
            round(
                height *
                width *
                bpp
            )
        )

        payload = generate_payload(
            target_bits,
            seed=2026 + int(bpp * 1000)
        )

        (
            direct_stego_ycrcb,
            embedding_map,
            embedded_bits,
            failed
        ) = embed_standard_ycrcb(
            cover_ycrcb,
            payload
        )

        # ----------------------------------------------------
        # TEST 1: DIRECT YCrCb EXTRACTION
        # ----------------------------------------------------

        direct_extracted, direct_details = (
            extract_from_ycrcb(
                direct_stego_ycrcb,
                embedding_map
            )
        )

        secret_bits = payload

        direct_ber, direct_errors = (
            calculate_ber(
                secret_bits[:embedded_bits],
                direct_extracted
            )
        )

        # ----------------------------------------------------
        # TEST 2: RGB ROUND-TRIP
        # ----------------------------------------------------

        stego_rgb = ycrcb_to_rgb(
            direct_stego_ycrcb
        )

        roundtrip_ycrcb = rgb_to_ycrcb(
            stego_rgb
        )

        extracted_roundtrip_bits, round_details = (
            extract_from_ycrcb(
                roundtrip_ycrcb,
                embedding_map
            )
        )

        roundtrip_ber, roundtrip_errors = calculate_ber(
            secret_bits[:embedded_bits],
            extracted_roundtrip_bits
        )

        (
            changed_pixels,
            changed_pairs,
            total_pairs,
            max_y_change,
            range_changed_pairs
        ) = compare_pair_differences(
            direct_stego_ycrcb,
            roundtrip_ycrcb,
            embedding_map
        )

        direct_exact = (
            len(direct_extracted) == embedded_bits
            and direct_extracted == secret_bits[:embedded_bits]
        )

        round_exact = (
            len(extracted_roundtrip_bits) == embedded_bits
            and extracted_roundtrip_bits == secret_bits[:embedded_bits]
        )

        print(
            f"Target bits                    : "
            f"{target_bits}"
        )

        print(
            f"Embedded bits                  : "
            f"{embedded_bits}"
        )

        print(
            f"Successful pairs               : "
            f"{len(embedding_map)}"
        )

        print(
            f"Embedding failures             : "
            f"{failed}"
        )

        print(
            "\nDIRECT YCrCb EXTRACTION"
        )

        print(
            f"  Extracted bits               : "
            f"{len(direct_extracted)}"
        )

        print(
            f"  BER                          : "
            f"{direct_ber:.8f}"
        )

        print(
            f"  Bit errors                   : "
            f"{direct_errors}"
        )

        print(
            f"  Exact extraction             : "
            f"{direct_exact}"
        )

        print(
            "\nAFTER YCrCb -> RGB -> YCrCb"
        )

        print(
            f"  Extracted bits               : "
            f"{len(extracted_roundtrip_bits)}"
        )

        print(
            f"Round-trip BER: {roundtrip_ber:.8f}, "
            f"Errors: {roundtrip_errors}"
        )

        print(
            f"  Exact extraction             : "
            f"{round_exact}"
        )

        print(
            "\nCOLOR-SPACE ROUND-TRIP EFFECT"
        )

        print(
            f"  Changed Y pixels             : "
            f"{changed_pixels}"
        )

        print(
            f"  Changed embedded pairs       : "
            f"{changed_pairs} / {total_pairs}"
        )

        print(
            f"  Range/capacity changes       : "
            f"{range_changed_pairs}"
        )

        print(
            f"  Maximum Y change             : "
            f"{max_y_change}"
        )

        # ----------------------------------------------------
        # Save pair-level diagnostic
        # ----------------------------------------------------

        import pandas as pd

        rows = []

        for d1, d2 in zip(
            direct_details,
            round_details
        ):

            rows.append({
                "row": d1["row"],
                "col": d1["col"],
                "bits": d1["bits"],
                "native_bits": d1["native_bits"],
                "embedded_difference": d1[
                    "embedded_difference"
                ],
                "direct_difference": d1[
                    "read_difference"
                ],
                "roundtrip_difference": d2[
                    "read_difference"
                ],
                "direct_extracted": d1.get(
                    "extracted",
                    ""
                ),
                "roundtrip_extracted": d2.get(
                    "extracted",
                    ""
                ),
                "expected": "".join(
                    str(x)
                    for x in payload[
                        d1["payload_start"]:
                        d1["payload_end"]
                    ]
                )
            })

        csv_path = os.path.join(
            RESULT_DIR,
            f"standard_ycrcb_diagnostic_"
            f"{bpp:.2f}.csv"
        )

        pd.DataFrame(rows).to_csv(
            csv_path,
            index=False
        )

        print(
            f"\nDiagnostic CSV: {csv_path}"
        )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        "\nInterpretation:"
    )

    print(
        "If Direct BER is 0 but Round-trip BER is > 0, "
        "the main error source is the RGB <-> YCrCb conversion."
    )


if __name__ == "__main__":
    main()
