import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from epvd import (
    RANGES,
    get_range,
    get_bit_capacity,
    pixel_difference,
    adjust_pixel_pair,
    embed_pair_adaptive,
    extract_pair,
    embed_pair,
)


def main():
    print("=" * 60)
    print("Testing Adaptive EPVD (5 to 8 Bits)")
    print("=" * 60)
    print()

    # ========================================================
    # 1. Single Pixel Pair Test (5, 6, 7, 8 bits)
    # ========================================================
    p1 = 100
    p2 = 145
    diff = pixel_difference(p1, p2)
    native = get_bit_capacity(diff)

    print("--- Test 1: Single Pair Evaluation ---")
    print(f"Original pixels: ({p1}, {p2})")
    print(f"Original difference: {diff}")
    print(f"Native bit capacity: {native} bits")
    print()

    for requested_bits in [5, 6, 7, 8]:
        secret_value = (2 ** requested_bits) - 1
        secret_bits = format(secret_value, f"0{requested_bits}b")

        print(f"Testing {requested_bits}-bit embedding:")
        print(f"  Secret payload:    {secret_bits}")

        new_p1, new_p2, actual_bits, success = embed_pair_adaptive(
            p1,
            p2,
            secret_bits,
            requested_bits,
        )

        print(f"  Stego pixels:      ({new_p1}, {new_p2})")
        print(f"  New difference:    {pixel_difference(new_p1, new_p2)}")
        print(f"  Actual bits:       {actual_bits}")
        print(f"  Success:           {success}")

        if success:
            extracted_bits, extracted_count = extract_pair(
                new_p1,
                new_p2,
                requested_bits,
            )
            expected_extracted = secret_bits[:actual_bits]
            match = extracted_bits == expected_extracted

            print(f"  Extracted bits:    {extracted_bits}")
            print(f"  Match:             {match}")
        print()

    # ========================================================
    # 2. Difference Ranges Test
    # ========================================================
    print("=" * 60)
    print("--- Test 2: Testing Across All EPVD Ranges ---")
    print("=" * 60)
    print()

    test_pairs = [
        ("Range 1 (0-31)", 100, 115, 5),
        ("Range 2 (32-63)", 100, 145, 5),
        ("Range 3 (64-127)", 50, 130, 6),
        ("Range 4 (128-255)", 20, 170, 7),
    ]

    for name, tp1, tp2, target_req in test_pairs:
        t_diff = pixel_difference(tp1, tp2)
        lower, upper, width = get_range(t_diff)
        cap = get_bit_capacity(t_diff)

        print(f"[{name}] Pixels: ({tp1}, {tp2}), Diff: {t_diff} in [{lower}, {upper}], Capacity: {cap} bits")

        for req in [5, 6, 7, 8]:
            # Generate alternating test bits '10101...'
            test_payload = ("10" * 4)[:req]

            stego_p1, stego_p2, actual, ok = embed_pair_adaptive(
                tp1, tp2, test_payload, req
            )

            if not ok:
                print(f"  Req {req}b -> Embedding Failed")
                continue

            extracted, _ = extract_pair(stego_p1, stego_p2, req)
            expected = test_payload[:actual]
            matched = extracted == expected

            in_range = lower <= pixel_difference(stego_p1, stego_p2) <= upper
            valid_pixels = 0 <= stego_p1 <= 255 and 0 <= stego_p2 <= 255

            status = "PASS" if (matched and in_range and valid_pixels) else "FAIL"
            print(
                f"  Req {req}b -> Embedded: {actual}b | Stego: ({stego_p1}, {stego_p2}) "
                f"| Diff: {pixel_difference(stego_p1, stego_p2)} | Match: {matched} [{status}]"
            )
        print()

    # ========================================================
    # 3. Policy Constraints & Edge Cases
    # ========================================================
    print("=" * 60)
    print("--- Test 3: Constraint Validation & Error Handling ---")
    print("=" * 60)
    print()

    for invalid_bits in [2, 3, 4, 9]:
        try:
            embed_pair_adaptive(100, 145, "1111", invalid_bits)
            print(f"  Invalid bits ({invalid_bits}): FAIL (Expected ValueError)")
        except ValueError as e:
            print(f"  Invalid bits ({invalid_bits}): Correctly rejected with ValueError -> '{e}'")

    print()
    print("=" * 60)
    print("All EPVD 5-to-8 Bit tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()