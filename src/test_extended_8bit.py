import os
import sys
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from extended_epvd import (
    embed_pair_8bit,
    extract_pair_8bit,
    pair_distortion
)


def generate_secret_bits(
    length,
    seed=2026
):

    rng = np.random.default_rng(
        seed
    )

    return "".join(
        rng.choice(
            ["0", "1"],
            size=length
        )
    )


def calculate_ber(
    original,
    extracted
):

    errors = sum(
        a != b
        for a, b in zip(
            original,
            extracted
        )
    )

    if len(original) == 0:
        return 0.0, 0

    return (
        errors / len(original),
        errors
    )


def main():

    print("=" * 70)
    print(
        "EXTENDED 8-BIT PAIR-STATE TEST"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # RANDOM TEST
    # ------------------------------------------------------

    rng = np.random.default_rng(
        2026
    )

    total_pairs = 10000

    original_bits = generate_secret_bits(
        total_pairs * 8
    )

    extracted_bits = []

    total_distortion = 0

    successful = 0

    for i in range(
        total_pairs
    ):

        p1 = int(
            rng.integers(
                0,
                256
            )
        )

        p2 = int(
            rng.integers(
                0,
                256
            )
        )

        start = i * 8

        payload = original_bits[
            start:start + 8
        ]

        (
            new_p1,
            new_p2,
            embedded,
            success
        ) = embed_pair_8bit(
            p1,
            p2,
            payload
        )

        if success:

            bits, extracted = (
                extract_pair_8bit(
                    new_p1,
                    new_p2
                )
            )

            extracted_bits.append(
                bits
            )

            total_distortion += (
                pair_distortion(
                    p1,
                    p2,
                    new_p1,
                    new_p2
                )
            )

            successful += 1

    extracted_bits = "".join(
        extracted_bits
    )

    ber, errors = calculate_ber(
        original_bits,
        extracted_bits
    )

    average_distortion = (
        total_distortion /
        successful
    )

    print(
        f"\nTest pairs              : "
        f"{total_pairs}"
    )

    print(
        f"Successful pairs        : "
        f"{successful}"
    )

    print(
        f"Embedded bits           : "
        f"{successful * 8}"
    )

    print(
        f"Extracted bits          : "
        f"{len(extracted_bits)}"
    )

    print(
        f"BER                     : "
        f"{ber:.10f}"
    )

    print(
        f"Bit errors              : "
        f"{errors}"
    )

    print(
        f"Average pair distortion : "
        f"{average_distortion:.4f}"
    )

    print(
        f"Exact extraction        : "
        f"{ber == 0}"
    )

    # ------------------------------------------------------
    # ALL 256 STATES TEST
    # ------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TESTING ALL 256 PAYLOAD STATES"
    )

    all_states_correct = True

    for state in range(256):

        bits = format(
            state,
            "08b"
        )

        p1, p2, embedded, success = (
            embed_pair_8bit(
                100,
                100,
                bits
            )
        )

        extracted, extracted_count = (
            extract_pair_8bit(
                p1,
                p2
            )
        )

        if (
            not success
            or embedded != 8
            or extracted != bits
            or extracted_count != 8
        ):

            all_states_correct = False

            print(
                f"FAILED STATE: {state}"
            )

            break

    print(
        f"All 256 states correct : "
        f"{all_states_correct}"
    )

    print(
        "\n" + "=" * 70
    )

    if (
        ber == 0
        and all_states_correct
    ):

        print(
            "8-BIT EXTENDED MODE PASSED"
        )

    else:

        print(
            "8-BIT EXTENDED MODE FAILED"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()