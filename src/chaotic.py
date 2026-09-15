import hashlib
import numpy as np


# =========================================================
# KEY → NUMERIC SEED
# =========================================================

def key_to_seed(secret_key):

    if not isinstance(secret_key, str):
        raise TypeError(
            "Secret key must be a string."
        )

    if len(secret_key) == 0:
        raise ValueError(
            "Secret key cannot be empty."
        )

    digest = hashlib.sha256(
        secret_key.encode("utf-8")
    ).hexdigest()

    integer_value = int(
        digest[:16],
        16
    )

    seed = (
        integer_value
        / float(0xFFFFFFFFFFFFFFFF)
    )

    seed = min(
        max(seed, 1e-10),
        1 - 1e-10
    )

    return seed


# =========================================================
# LOGISTIC MAP
# =========================================================

def logistic_map(
    seed,
    length,
    r=3.99
):

    if not (0 < seed < 1):

        raise ValueError(
            "Seed must be between 0 and 1."
        )

    sequence = np.zeros(
        length,
        dtype=np.float64
    )

    x = seed

    for i in range(length):

        x = r * x * (1 - x)

        sequence[i] = x

    return sequence


# =========================================================
# CHAOTIC ORDER
# =========================================================

def generate_chaotic_order(
    secret_key,
    total_positions
):

    seed = key_to_seed(
        secret_key
    )

    sequence = logistic_map(
        seed,
        total_positions
    )

    order = np.argsort(
        sequence
    )

    return order


# =========================================================
# CHAOTIC BLOCK ORDER
# =========================================================

def generate_block_order(
    total_blocks,
    secret_key
):

    return generate_chaotic_order(
        secret_key,
        total_blocks
    )


# =========================================================
# CHAOTIC PIXEL-PAIR POSITIONS
# =========================================================

def generate_pair_positions(
    height,
    width,
    secret_key
):

    positions = []

    for row in range(height):

        for col in range(
            0,
            width - 1,
            2
        ):

            positions.append(
                (
                    row,
                    col,
                    row,
                    col + 1
                )
            )

    order = generate_chaotic_order(
        secret_key,
        len(positions)
    )

    return [
        positions[index]
        for index in order
    ]