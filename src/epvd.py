import math


# ============================================================
# EPVD Difference Ranges
# ============================================================

RANGES = [
    (0, 31),
    (32, 63),
    (64, 127),
    (128, 255)
]


# ============================================================
# Difference Range
# ============================================================

def get_range(difference):

    difference = int(difference)

    for lower, upper in RANGES:
        if lower <= difference <= upper:
            return lower, upper, upper - lower + 1

    raise ValueError(
        f"Invalid pixel difference: {difference}"
    )


# ============================================================
# Native Capacity
# ============================================================

def get_bit_capacity(difference):
    """
    Native number of bits available for a difference value.

    0-31     -> 5 bits
    32-63    -> 5 bits
    64-127   -> 6 bits
    128-255  -> 7 bits
    """

    lower, upper, width = get_range(difference)

    return int(math.floor(math.log2(width)))


# ============================================================
# Pixel Difference
# ============================================================

def pixel_difference(p1, p2):
    return abs(int(p2) - int(p1))


# ============================================================
# Find Feasible Pixel Pair
# ============================================================

def adjust_pixel_pair(p1, p2, target_difference):

    p1 = int(p1)
    p2 = int(p2)
    target_difference = int(target_difference)

    # Preserve original direction
    if p2 >= p1:

        # Try distributing the change approximately equally
        delta = target_difference - (p2 - p1)

        new_p1 = p1 - delta // 2
        new_p2 = p2 + (delta - delta // 2)

        if (
            0 <= new_p1 <= 255
            and
            0 <= new_p2 <= 255
            and
            abs(new_p2 - new_p1) == target_difference
        ):
            return new_p1, new_p2, True

        # Search around the original pair for a feasible pair
        center = (p1 + p2) / 2.0

        min_p1 = max(0, int(math.ceil(center - target_difference / 2)))
        max_p1 = min(
            255 - target_difference,
            int(math.floor(center - target_difference / 2)) + 10
        )

        for candidate_p1 in range(
            max(0, min_p1 - 10),
            min(255 - target_difference, max_p1 + 10) + 1
        ):
            candidate_p2 = candidate_p1 + target_difference

            if (
                0 <= candidate_p2 <= 255
                and
                abs(
                    (candidate_p2 - candidate_p1)
                    - target_difference
                ) == 0
            ):
                # Choose pair closest to original center
                if abs(
                    ((candidate_p1 + candidate_p2) / 2)
                    - center
                ) <= target_difference + 10:
                    return candidate_p1, candidate_p2, True

    else:

        delta = target_difference - (p1 - p2)

        new_p1 = p1 + delta // 2
        new_p2 = p2 - (delta - delta // 2)

        if (
            0 <= new_p1 <= 255
            and
            0 <= new_p2 <= 255
            and
            abs(new_p2 - new_p1) == target_difference
        ):
            return new_p1, new_p2, True

        center = (p1 + p2) / 2.0

        for candidate_p2 in range(0, 256 - target_difference):

            candidate_p1 = candidate_p2 + target_difference

            if (
                0 <= candidate_p1 <= 255
                and
                0 <= candidate_p2 <= 255
            ):
                if abs(
                    ((candidate_p1 + candidate_p2) / 2)
                    - center
                ) <= target_difference + 10:

                    return candidate_p1, candidate_p2, True

    return p1, p2, False


# ============================================================
# Adaptive EPVD Embedding
# ============================================================

def embed_pair_adaptive(
    p1,
    p2,
    secret_bits,
    requested_bits
):

    requested_bits = int(requested_bits)

    if requested_bits not in range(1, 9):
        raise ValueError(
            "requested_bits must be between 1 and 8"
        )

    difference = pixel_difference(p1, p2)

    lower, upper, range_width = get_range(
        difference
    )

    native_bits = get_bit_capacity(
        difference
    )

    actual_bits = min(
        requested_bits,
        native_bits,
        len(secret_bits)
    )

    if actual_bits <= 0:
        return p1, p2, 0, False

    payload = secret_bits[:actual_bits]

    payload_value = int(
        payload,
        2
    )

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------

    levels = 2 ** actual_bits

    step = range_width / levels

    target_difference = int(
        lower
        +
        payload_value * step
        +
        step / 2
    )

    target_difference = max(
        lower,
        min(upper, target_difference)
    )

    # --------------------------------------------------------
    # Boundary-aware adjustment
    # --------------------------------------------------------

    new_p1, new_p2, success = adjust_pixel_pair(
        p1,
        p2,
        target_difference
    )

    if not success:
        return p1, p2, 0, False

    # --------------------------------------------------------
    # Verify resulting difference
    # --------------------------------------------------------

    new_difference = pixel_difference(
        new_p1,
        new_p2
    )

    if not (
        lower
        <= new_difference
        <= upper
    ):
        return p1, p2, 0, False

    return (
        new_p1,
        new_p2,
        actual_bits,
        True
    )


# ============================================================
# Extraction
# ============================================================

def extract_pair(
    p1,
    p2,
    expected_bits=None
):

    difference = pixel_difference(
        p1,
        p2
    )

    lower, upper, range_width = get_range(
        difference
    )

    native_bits = get_bit_capacity(
        difference
    )

    if expected_bits is None:
        actual_bits = native_bits
    else:
        actual_bits = min(
            int(expected_bits),
            native_bits
        )

    levels = 2 ** actual_bits

    step = range_width / levels

    value = difference - lower

    payload_value = int(
        value / step
    )

    payload_value = max(
        0,
        min(
            levels - 1,
            payload_value
        )
    )

    bits = format(
        payload_value,
        f"0{actual_bits}b"
    )

    return bits, actual_bits


# ============================================================
# Standard EPVD
# ============================================================

def embed_pair(
    p1,
    p2,
    secret_bits
):

    difference = pixel_difference(
        p1,
        p2
    )

    native_bits = get_bit_capacity(
        difference
    )

    if len(secret_bits) < native_bits:
        return p1, p2, 0, False

    return embed_pair_adaptive(
        p1,
        p2,
        secret_bits,
        native_bits
    )