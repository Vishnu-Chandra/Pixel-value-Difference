import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "secret"
)

SECRET_PATH = os.path.join(
    SECRET_DIR,
    "secret.txt"
)

os.makedirs(SECRET_DIR, exist_ok=True)

# 64 KB = 65,536 bytes
SIZE = 65536

# Deterministic pseudo-random data.
# This gives us reproducible experiments.
data = bytes(
    ((i * 37 + 91) % 256)
    for i in range(SIZE)
)

with open(SECRET_PATH, "wb") as f:
    f.write(data)

print("=" * 60)
print("SECRET DATA GENERATED")
print("=" * 60)
print(f"File       : {SECRET_PATH}")
print(f"Size       : {SIZE} bytes")
print(f"Bits       : {SIZE * 8}")
print("=" * 60)