import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "features.csv")


def get_capacity(difference):

    if 0 <= difference <= 31:
        return 5

    elif 32 <= difference <= 63:
        return 5

    elif 64 <= difference <= 127:
        return 6

    elif 128 <= difference <= 255:
        return 7

    return 0


# Load dataset
df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

# Calculate EPVD capacity
df["capacity"] = df["pixel_difference"].apply(get_capacity)


print("\nOverall capacity distribution:")
print(df["capacity"].value_counts().sort_index())


print("\nCapacity percentage:")
print(
    (df["capacity"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)


print("\nPixel difference statistics:")
print(df["pixel_difference"].describe())


print("\nCapacity distribution by image:")

for image_name, group in df.groupby("image_name"):

    print("\n", image_name)

    print(
        group["capacity"]
        .value_counts()
        .sort_index()
    )


print("\nPixel difference range:")
print(
    "Minimum:",
    df["pixel_difference"].min()
)

print(
    "Maximum:",
    df["pixel_difference"].max()
)