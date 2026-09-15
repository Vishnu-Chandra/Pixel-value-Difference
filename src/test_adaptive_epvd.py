import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from epvd import (
    embed_pair_adaptive,
    extract_pair
)


print("----------------------------------------")
print("Testing Adaptive EPVD")
print("----------------------------------------")
print()


p1 = 100
p2 = 145

print(
    "Original pixels:",
    p1,
    p2
)

print(
    "Original difference:",
    abs(p2 - p1)
)

print()


for requested_bits in [2, 3, 4, 5]:

    secret_value = (
        2 ** requested_bits
    ) - 1

    secret_bits = format(
        secret_value,
        f"0{requested_bits}b"
    )

    print(
        f"Testing {requested_bits}-bit embedding"
    )

    new_p1, new_p2, actual_bits, success = (
        embed_pair_adaptive(
            p1,
            p2,
            secret_bits,
            requested_bits
        )
    )

    print(
        "  Secret:",
        secret_bits
    )

    print(
        "  New pixels:",
        new_p1,
        new_p2
    )

    print(
        "  Actual bits:",
        actual_bits
    )

    print(
        "  Success:",
        success
    )

    if success:

        extracted_bits, extracted_count = (
            extract_pair(
                new_p1,
                new_p2,
                requested_bits
            )
        )

        print(
            "  Extracted:",
            extracted_bits
        )

        print(
            "  Match:",
            extracted_bits == secret_bits
        )

    print()