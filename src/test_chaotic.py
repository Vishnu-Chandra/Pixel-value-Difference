import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from chaotic import (
    key_to_seed,
    logistic_map,
    generate_pair_positions
)


SECRET_KEY = "AI_EPVD_2026"


print("----------------------------------------")
print("Testing Chaotic Pixel Selection")
print("----------------------------------------")
print()


# =========================================================
# KEY → SEED
# =========================================================

seed = key_to_seed(
    SECRET_KEY
)

print(
    "Secret key:",
    SECRET_KEY
)

print(
    "Generated seed:",
    seed
)

print()


# =========================================================
# LOGISTIC MAP
# =========================================================

sequence = logistic_map(
    seed,
    10
)

print(
    "First 10 chaotic values:"
)

for value in sequence:

    print(
        f"{value:.10f}"
    )

print()


# =========================================================
# PIXEL PAIR ORDER
# =========================================================

positions = generate_pair_positions(
    8,
    8,
    SECRET_KEY
)

print(
    "Total pixel pairs:",
    len(positions)
)

print()

print(
    "First 10 chaotic positions:"
)

for position in positions[:10]:

    print(
        position
    )

print()

print(
    "Chaotic pixel selection test completed."
)