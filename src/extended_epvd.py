# src/extended_epvd.py

import itertools

BITS_PER_PAIR = 8
NUM_STATES = 256


def pair_to_state(p1, p2):
    """
    Map an ordered pixel pair to one of 256 states.

    The state is based on the relative difference between the pixels.
    """

    p1 = int(p1)
    p2 = int(p2)

    # Encode the signed difference into 0..255.
    difference = p2 - p1

    return difference + 255


def state_to_difference(state):
    """
    Convert state back to signed difference.
    """

    state = int(state)

    if state < 0 or state >= NUM_STATES:
        raise ValueError("Invalid state")

    return state - 255


def find_best_pair(p1, p2, target_state):
    """
    Find the closest valid pair having the desired signed difference.

    We minimize squared distortion:

        (P1' - P1)^2 + (P2' - P2)^2

    """

    p1 = int(p1)
    p2 = int(p2)

    target_difference = state_to_difference(target_state)

    best_p1 = None
    best_p2 = None
    best_distortion = float("inf")

    # For a fixed difference d:
    #
    # p2' = p1' + d
    #
    # Search around the original pair's midpoint.

    midpoint = (p1 + p2) / 2.0

    if target_difference >= 0:

        low_p1 = 0
        high_p1 = 255 - target_difference

    else:

        low_p1 = -target_difference
        high_p1 = 255

    # Ideal first pixel that preserves the original midpoint.
    ideal_p1 = midpoint - target_difference / 2.0

    candidates = set()

    # Search around the ideal point.
    center = int(round(ideal_p1))

    for offset in range(-10, 11):

        candidate = center + offset

        if low_p1 <= candidate <= high_p1:
            candidates.add(candidate)

    # Always include boundaries.
    candidates.add(low_p1)
    candidates.add(high_p1)

    for new_p1 in candidates:

        new_p2 = new_p1 + target_difference

        if not (0 <= new_p1 <= 255):
            continue

        if not (0 <= new_p2 <= 255):
            continue

        distortion = (
            (new_p1 - p1) ** 2 +
            (new_p2 - p2) ** 2
        )

        if distortion < best_distortion:

            best_distortion = distortion
            best_p1 = new_p1
            best_p2 = new_p2

    if best_p1 is None:

        return p1, p2, False, float("inf")

    return (
        best_p1,
        best_p2,
        True,
        best_distortion
    )


def embed_pair_8bit(p1, p2, secret_bits):

    if len(secret_bits) < 8:

        return p1, p2, 0, False

    secret_bits = secret_bits[:8]

    target_state = int(secret_bits, 2)

    new_p1, new_p2, success, distortion = find_best_pair(
        p1,
        p2,
        target_state
    )

    if not success:

        return p1, p2, 0, False

    return (
        new_p1,
        new_p2,
        8,
        True
    )


def extract_pair_8bit(p1, p2):

    state = pair_to_state(p1, p2)

    # State is guaranteed to be 0..255.
    bits = format(state, "08b")

    return bits, 8


def pair_distortion(p1, p2, new_p1, new_p2):

    return (
        (int(new_p1) - int(p1)) ** 2 +
        (int(new_p2) - int(p2)) ** 2
    )


def test_all_states():

    errors = []

    # Test states using representative pairs.
    for state in range(NUM_STATES):

        target_difference = state_to_difference(state)

        # Choose a safe original pair near the middle.
        if target_difference >= 0:

            p1 = 128 - target_difference // 2
            p2 = p1 + target_difference

        else:

            p2 = 128 + target_difference // 2
            p1 = p2 - target_difference

        # Clamp.
        p1 = max(0, min(255, p1))
        p2 = max(0, min(255, p2))

        bits = format(state, "08b")

        new_p1, new_p2, embedded, success = embed_pair_8bit(
            p1,
            p2,
            bits
        )

        if not success:

            errors.append(
                (state, "embedding failed")
            )
            continue

        extracted, count = extract_pair_8bit(
            new_p1,
            new_p2
        )

        if extracted != bits:

            errors.append(
                (
                    state,
                    bits,
                    extracted,
                    new_p1,
                    new_p2
                )
            )

    return errors