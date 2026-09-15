import os
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "features.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "labeled_features.csv"
)


# =========================================================
# FEATURE WEIGHTS
# =========================================================

WEIGHTS = {
    "variance": 0.20,
    "entropy": 0.15,
    "edge_density": 0.25,
    "gradient_magnitude": 0.20,
    "mean_pixel_difference": 0.10,
    "max_pixel_difference": 0.05,
    "std_pixel_difference": 0.05
}


# =========================================================
# MIN-MAX NORMALIZATION
# =========================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# =========================================================
# GENERATE ADAPTIVE LABELS
# =========================================================

def generate_labels():

    print("----------------------------------------")
    print("Generating Texture-Adaptive Labels")
    print("----------------------------------------")
    print()

    df = pd.read_csv(
        FEATURE_FILE
    )

    # -----------------------------------------------------
    # Normalize all seven features
    # -----------------------------------------------------

    normalized_features = {}

    for feature in WEIGHTS:

        normalized_features[feature] = normalize(
            df[feature]
        )

    # -----------------------------------------------------
    # Calculate texture suitability score
    # -----------------------------------------------------

    texture_score = np.zeros(
        len(df)
    )

    for feature, weight in WEIGHTS.items():

        texture_score += (
            normalized_features[feature]
            * weight
        )

    df["texture_score"] = texture_score

    # -----------------------------------------------------
    # Use dataset percentiles to divide blocks
    # into four adaptive capacity groups.
    # -----------------------------------------------------

    q25 = df["texture_score"].quantile(
        0.25
    )

    q50 = df["texture_score"].quantile(
        0.50
    )

    q75 = df["texture_score"].quantile(
        0.75
    )

    def assign_label(score):

        if score <= q25:
            return 2

        elif score <= q50:
            return 3

        elif score <= q75:
            return 4

        else:
            return 5

    df["embedding_bits"] = (
        df["texture_score"]
        .apply(assign_label)
    )

    # -----------------------------------------------------
    # Remove temporary score before training
    # -----------------------------------------------------

    df = df.drop(
        columns=["texture_score"]
    )

    # -----------------------------------------------------
    # Save dataset
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print(
        "Texture score thresholds:"
    )

    print(
        f"25th percentile: {q25:.6f}"
    )

    print(
        f"50th percentile: {q50:.6f}"
    )

    print(
        f"75th percentile: {q75:.6f}"
    )

    print()

    print(
        "Overall label distribution:"
    )

    distribution = (
        df["embedding_bits"]
        .value_counts()
        .sort_index()
    )

    print(
        distribution
    )

    print()

    print(
        "Percentage distribution:"
    )

    percentages = (
        df["embedding_bits"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        percentages.round(2)
    )

    print()

    print(
        "Dataset shape:",
        df.shape
    )

    print()

    print(
        "Columns:"
    )

    print(
        list(df.columns)
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    generate_labels()