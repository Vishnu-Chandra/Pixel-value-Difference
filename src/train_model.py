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
# Find Difference Range
# ============================================================

def get_range(difference):
    """
    Return:
        lower_bound,
        upper_bound,
        range_width
    """

    for lower, upper in RANGES:
        if lower <= difference <= upper:
            return lower, upper, upper - lower + 1

    raise ValueError(f"Invalid pixel difference: {difference}")


# ============================================================
# Native EPVD Capacity
# ============================================================

def get_bit_capacity(difference):
    """
    Native number of bits available for a difference value.

    0–31     -> 5 bits
    32–63    -> 5 bits
    64–127   -> 6 bits
    128–255  -> 8 bits
    """

    lower, upper, width = get_range(difference)

    return int(math.floor(math.log2(width)))


# ============================================================
# Pixel Difference
# ============================================================

def pixel_difference(p1, p2):
    return abs(int(p2) - int(p1))


# ============================================================
# Adjust Pixel Pair
# ============================================================

def adjust_pixel_pair(p1, p2, target_difference):
    """
    Adjust a pixel pair so that its difference becomes
    target_difference while preserving the original direction.

    Returns:
        new_p1, new_p2, success
    """

    p1 = int(p1)
    p2 = int(p2)
    target_difference = int(target_difference)

    current_difference = abs(p2 - p1)

    # Already correct
    if current_difference == target_difference:
        return p1, p2, True

    # Preserve original direction
    if p2 >= p1:
        center = (p1 + p2) / 2.0

        new_p1 = int(round(center - target_difference / 2.0))
        new_p2 = int(round(center + target_difference / 2.0))

    else:
        center = (p1 + p2) / 2.0

        new_p1 = int(round(center + target_difference / 2.0))
        new_p2 = int(round(center - target_difference / 2.0))

    # --------------------------------------------------------
    # Boundary correction
    # --------------------------------------------------------

    if new_p1 < 0:
        shift = -new_p1
        new_p1 += shift
        new_p2 += shift

    if new_p2 < 0:
        shift = -new_p2
        new_p1 += shift
        new_p2 += shift

    if new_p1 > 255:
        shift = new_p1 - 255
        new_p1 -= shift
        new_p2 -= shift

    if new_p2 > 255:
        shift = new_p2 - 255
        new_p1 -= shift
        new_p2 -= shift

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not (0 <= new_p1 <= 255 and 0 <= new_p2 <= 255):
        return p1, p2, False

    final_difference = abs(new_p2 - new_p1)

    if final_difference != target_difference:
        return p1, p2, False

    return new_p1, new_p2, True


# ============================================================
# Adaptive Pair Embedding
# ============================================================

def embed_pair_adaptive(p1, p2, secret_bits, requested_bits):
    """
    Embed secret_bits into a pixel pair.

    requested_bits:
        5, 6, 7, or 8

    The actual number of embedded bits is limited by
    the native capacity of the original difference range.

    Returns:
        new_p1,
        new_p2,
        actual_bits,
        success
    """

    requested_bits = int(requested_bits)

    if requested_bits not in [5, 6, 7, 8]:
        raise ValueError(
            "requested_bits must be one of: 5, 6, 7, 8"
        )

    p1 = int(p1)
    p2 = int(p2)

    difference = pixel_difference(p1, p2)

    lower, upper, range_width = get_range(difference)

    native_bits = get_bit_capacity(difference)

    actual_bits = min(
        requested_bits,
        native_bits
    )

    if actual_bits <= 0:
        return p1, p2, 0, False

    # --------------------------------------------------------
    # Convert secret bits to integer
    # --------------------------------------------------------

    if len(secret_bits) < actual_bits:
        return p1, p2, 0, False

    payload = secret_bits[:actual_bits]

    payload_value = int(payload, 2)

    # --------------------------------------------------------
    # Divide difference range into quantization intervals
    # --------------------------------------------------------

    levels = 2 ** actual_bits

    step = range_width / levels

    target_difference = int(
        lower +
        payload_value * step +
        step / 2.0
    )

    # Clamp target
    target_difference = max(
        lower,
        min(upper, target_difference)
    )

    # --------------------------------------------------------
    # Adjust pair
    # --------------------------------------------------------

    new_p1, new_p2, success = adjust_pixel_pair(
        p1,
        p2,
        target_difference
    )

    if not success:
        return p1, p2, 0, False

    # --------------------------------------------------------
    # Verify that the resulting difference remains
    # inside the intended range
    # --------------------------------------------------------

    new_difference = pixel_difference(
        new_p1,
        new_p2
    )

    if not (lower <= new_difference <= upper):
        return p1, p2, 0, False

    return (
        new_p1,
        new_p2,
        actual_bits,
        True
    )


# ============================================================
# Adaptive Pair Extraction
# ============================================================

def extract_pair(p1, p2, expected_bits=None):
    """
    Extract the payload from an EPVD pixel pair.

    expected_bits should normally be the same number
    predicted during embedding.
    """

    difference = pixel_difference(p1, p2)

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

    if actual_bits <= 0:
        return ""

    levels = 2 ** actual_bits

    step = range_width / levels

    value = difference - lower

    payload_value = int(
        value / step
    )

    # Prevent numerical edge cases
    payload_value = max(
        0,
        min(levels - 1, payload_value)
    )

    return format(
        payload_value,
        f"0{actual_bits}b"
    )


# ============================================================
# Backward-Compatible Standard EPVD
# ============================================================

def embed_pair(p1, p2, secret_bits):
    """
    Standard EPVD embedding.

    Automatically uses the native capacity of
    the original difference range.
    """

    difference = pixel_difference(p1, p2)

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