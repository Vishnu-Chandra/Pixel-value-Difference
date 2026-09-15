"""
STANDARD EPVD BER DIAGNOSTIC
----------------------------
Run from the src directory:

    python standard_ber_diagnostic.py

Purpose:
- Diagnose where Standard EPVD extraction errors occur.
- Tests Lena at 0.25, 0.50, 1.00 and 1.50 BPP.
- Reports embedded bits, extracted bits, BER, first error positions,
  errors by EPVD native capacity (5/6/7), and whether errors cluster
  near range/boundary conditions.
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
        raise FileNotFoundError(path)

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


def embed_standard_with_diagnostics(
    cover_rgb,
    payload_bits
):
    """
    Standard EPVD:
    native capacity is selected only from the original pair difference.
    """

    ycrcb = rgb_to_ycrcb(cover_rgb)
    stego_ycrcb = ycrcb.copy()

    original_y = ycrcb[:, :, 0]
    stego_y = stego_ycrcb[:, :, 0]

    height, width = original_y.shape

    payload_position = 0
    embedding_map = []

    failed = 0

    for row in range(height):

        for col in range(0, width - 1, 2):

            if payload_position >= len(payload_bits):
                break

            p1 = int(original_y[row, col])
            p2 = int(original_y[row, col + 1])

            original_difference = pixel_difference(
                p1,
                p2
            )

            native_bits = get_bit_capacity(
                original_difference
            )

            remaining = (
                len(payload_bits)
                -
                payload_position
            )

            bits_to_embed = min(
                native_bits,
                remaining
            )

            chunk = payload_bits[
                payload_position:
                payload_position + bits_to_embed
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

            stego_y[row, col] = np.uint8(
                np.clip(new_p1, 0, 255)
            )

            stego_y[row, col + 1] = np.uint8(
                np.clip(new_p2, 0, 255)
            )

            new_difference = pixel_difference(
                new_p1,
                new_p2
            )

            embedding_map.append({
                "row": row,
                "col": col,
                "native_bits": native_bits,
                "actual_bits": actual_bits,
                "original_difference": original_difference,
                "new_difference": new_difference,
                "original_p1": p1,
                "original_p2": p2,
                "new_p1": int(new_p1),
                "new_p2": int(new_p2),
                "payload_start": payload_position,
                "payload_end": (
                    payload_position + actual_bits
                )
            })

            payload_position += actual_bits

        if payload_position >= len(payload_bits):
            break

    stego_rgb = ycrcb_to_rgb(stego_ycrcb)

    return (
        stego_rgb,
        embedding_map,
        payload_position,
        failed
    )


def extract_with_diagnostics(
    stego_rgb,
    embedding_map,
    payload_bits
):

    ycrcb = rgb_to_ycrcb(stego_rgb)
    y = ycrcb[:, :, 0]

    extracted = []

    diagnostics = []

    for item in embedding_map:

        row = item["row"]
        col = item["col"]
        expected_bits = item["actual_bits"]

        p1 = int(y[row, col])
        p2 = int(y[row, col + 1])

        stego_difference = pixel_difference(
            p1,
            p2
        )

        try:
            bits, actual_bits = extract_pair(
                p1,
                p2,
                expected_bits
            )
        except Exception as error:
            diagnostics.append({
                **item,
                "stego_p1": p1,
                "stego_p2": p2,
                "stego_difference": stego_difference,
                "extracted_bits": "",
                "expected_bits": "",
                "error": str(error)
            })
            continue

        expected = payload_bits[
            item["payload_start"]:
            item["payload_end"]
        ]

        extracted_string = "".join(
            str(x) for x in bits[:expected_bits]
        )

        expected_string = "".join(
            str(x) for x in expected
        )

        diagnostics.append({
            **item,
            "stego_p1": p1,
            "stego_p2": p2,
            "stego_difference": stego_difference,
            "extracted_bits": extracted_string,
            "expected_bits": expected_string,
            "error": ""
        })

        extracted.extend(
            int(x) for x in bits[:expected_bits]
        )

    return extracted, diagnostics


def main():

    print("=" * 70)
    print("STANDARD EPVD BER DIAGNOSTIC")
    print("=" * 70)

    cover = load_rgb(COVER_PATH)

    print(
        f"\nCover: lena.png"
    )

    print(
        f"Resolution: "
        f"{cover.shape[1]} x {cover.shape[0]}"
    )

    all_rows = []

    for bpp in TEST_BPP:

        print("\n" + "-" * 70)
        print(f"Testing {bpp:.2f} BPP")
        print("-" * 70)

        height, width = cover.shape[:2]

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
            stego,
            embedding_map,
            embedded_bits,
            failed
        ) = embed_standard_with_diagnostics(
            cover,
            payload
        )

        extracted, diagnostics = (
            extract_with_diagnostics(
                stego,
                embedding_map,
                payload
            )
        )

        compare_n = min(
            len(payload),
            len(extracted)
        )

        errors = []

        for i in range(compare_n):
            if payload[i] != extracted[i]:
                errors.append(i)

        ber = (
            len(errors) / compare_n
            if compare_n > 0
            else 1.0
        )

        print(
            f"Target bits       : {target_bits}"
        )

        print(
            f"Embedded bits     : {embedded_bits}"
        )

        print(
            f"Extracted bits    : {len(extracted)}"
        )

        print(
            f"Successful pairs  : {len(embedding_map)}"
        )

        print(
            f"Failed pairs      : {failed}"
        )

        print(
            f"BER               : {ber:.8f}"
        )

        print(
            f"Bit errors        : {len(errors)}"
        )

        print(
            f"Extraction exact  : "
            f"{len(errors) == 0 and len(extracted) == len(payload)}"
        )

        if errors:
            print(
                "\nFirst 20 erroneous bit positions:"
            )

            print(errors[:20])

        # Error count grouped by native capacity
        capacity_stats = {}

        for item in diagnostics:

            expected = item["expected_bits"]
            extracted_bits = item["extracted_bits"]

            if not expected:
                continue

            local_errors = sum(
                a != b
                for a, b in zip(
                    expected,
                    extracted_bits
                )
            )

            capacity = item["native_bits"]

            if capacity not in capacity_stats:
                capacity_stats[capacity] = {
                    "pairs": 0,
                    "bits": 0,
                    "errors": 0
                }

            capacity_stats[capacity]["pairs"] += 1
            capacity_stats[capacity]["bits"] += len(expected)
            capacity_stats[capacity]["errors"] += local_errors

        print(
            "\nErrors by native EPVD capacity:"
        )

        for capacity in sorted(capacity_stats):

            s = capacity_stats[capacity]

            local_ber = (
                s["errors"] / s["bits"]
                if s["bits"] else 0
            )

            print(
                f"  {capacity} bits | "
                f"pairs={s['pairs']:<7} | "
                f"bits={s['bits']:<7} | "
                f"errors={s['errors']:<6} | "
                f"BER={local_ber:.8f}"
            )

        # Range preservation
        range_changes = 0

        for item in diagnostics:

            old_d = item["original_difference"]
            new_d = item["new_difference"]

            old_capacity = get_bit_capacity(old_d)

            new_capacity = get_bit_capacity(new_d)

            if old_capacity != new_capacity:
                range_changes += 1

        print(
            f"\nPairs changing native capacity range: "
            f"{range_changes}"
        )

        # Save diagnostic CSV
        csv_path = os.path.join(
            RESULT_DIR,
            f"standard_ber_diagnostic_{bpp:.2f}.csv"
        )

        import pandas as pd

        pd.DataFrame(
            diagnostics
        ).to_csv(
            csv_path,
            index=False
        )

        print(
            f"Diagnostic CSV: {csv_path}"
        )

        all_rows.extend(
            diagnostics
        )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        "\nFour diagnostic CSV files were generated "
        "in the results folder."
    )


if __name__ == "__main__":
    main()
