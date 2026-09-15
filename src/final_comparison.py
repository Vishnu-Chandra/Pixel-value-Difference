import os
import glob
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)


IMAGES = [
    "lena",
    "baboon",
    "car-house",
    "jet",
    "peppers",
    "sailboat"
]

BPP_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]


# ============================================================
# LOAD CSV FILES
# ============================================================

def load_method_files(method_prefix):

    frames = []

    for image in IMAGES:

        filename = (
            f"{method_prefix}_{image}.csv"
        )

        path = os.path.join(
            RESULTS_DIR,
            filename
        )

        if not os.path.exists(path):

            print(
                f"WARNING: Missing {filename}"
            )

            continue

        df = pd.read_csv(path)

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found for {method_prefix}"
        )

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# LOAD ALL METHODS
# ============================================================

print("=" * 80)
print("LOADING EXPERIMENTAL RESULTS")
print("=" * 80)

standard = load_method_files(
    "standard_final"
)

ai = load_method_files(
    "ai_final"
)

chaotic = load_method_files(
    "ai_chaotic_final"
)


standard["Method"] = "Standard EPVD"
ai["Method"] = "AI-EPVD"
chaotic["Method"] = "AI + Chaotic EPVD"


all_results = pd.concat(
    [
        standard,
        ai,
        chaotic
    ],
    ignore_index=True
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

if "Target_BPP" not in all_results.columns:
    all_results["Target_BPP"] = all_results["BPP"]
elif "BPP" in all_results.columns:
    all_results["Target_BPP"] = all_results["Target_BPP"].fillna(all_results["BPP"])


# ============================================================
# VERIFY EXPERIMENTS
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT COUNT")
print("=" * 80)

for method in [
    "Standard EPVD",
    "AI-EPVD",
    "AI + Chaotic EPVD"
]:

    count = len(
        all_results[
            all_results["Method"] == method
        ]
    )

    print(
        f"{method:25s}: {count} experiments"
    )


# ============================================================
# DIRECT EXTRACTION CHECK
# ============================================================

print("\n" + "=" * 80)
print("DIRECT YCrCb EXTRACTION CHECK")
print("=" * 80)

for method in [
    "Standard EPVD",
    "AI-EPVD",
    "AI + Chaotic EPVD"
]:

    subset = all_results[
        all_results["Method"] == method
    ]

    if "Direct_YCrCb_BER" in subset.columns:

        max_ber = subset[
            "Direct_YCrCb_BER"
        ].max()

        print(
            f"{method:25s}: "
            f"Maximum BER = {max_ber:.10f}"
        )

    if "Direct_Exact" in subset.columns:

        exact_count = (
            subset["Direct_Exact"]
            .astype(bool)
            .sum()
        )

        print(
            f"{'':25s}  "
            f"Exact = {exact_count}/{len(subset)}"
        )


# ============================================================
# AVERAGE BY BPP
# ============================================================

metrics = [
    "MSE",
    "PSNR",
    "SSIM",
    "Actual_BPP",
    "Direct_YCrCb_BER",
    "Roundtrip_RGB_BER"
]


available_metrics = [
    metric
    for metric in metrics
    if metric in all_results.columns
]


summary = (
    all_results
    .groupby(
        ["Target_BPP", "Method"]
    )[available_metrics]
    .mean()
    .reset_index()
)


# ============================================================
# THREE-WAY TABLE
# ============================================================

print("\n" + "=" * 80)
print("THREE-WAY PERFORMANCE COMPARISON")
print("=" * 80)


for bpp in BPP_LEVELS:

    print(
        f"\n{'-' * 80}"
    )

    print(
        f"BPP = {bpp:.2f}"
    )

    print(
        f"{'-' * 80}"
    )

    subset = summary[
        np.isclose(
            summary["Target_BPP"],
            bpp
        )
    ]

    display_columns = [
        "Method"
    ] + available_metrics

    print(
        subset[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )


# ============================================================
# OVERALL AVERAGES
# ============================================================

overall = (
    all_results
    .groupby("Method")[available_metrics]
    .mean()
    .reset_index()
)


print("\n" + "=" * 80)
print("OVERALL AVERAGE PERFORMANCE")
print("=" * 80)

print(
    overall.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

overall_indexed = overall.set_index(
    "Method"
)


def improvement(
    new_value,
    baseline_value
):

    if baseline_value == 0:
        return 0.0

    return (
        (new_value - baseline_value)
        / abs(baseline_value)
    ) * 100


standard_row = overall_indexed.loc[
    "Standard EPVD"
]

ai_row = overall_indexed.loc[
    "AI-EPVD"
]

chaotic_row = overall_indexed.loc[
    "AI + Chaotic EPVD"
]


print("\n" + "=" * 80)
print("AI-EPVD vs STANDARD EPVD")
print("=" * 80)


for metric in [
    "MSE",
    "PSNR",
    "SSIM",
    "Roundtrip_RGB_BER"
]:

    if metric not in overall_indexed.columns:
        continue

    change = improvement(
        ai_row[metric],
        standard_row[metric]
    )

    if metric == "MSE":
        interpretation = (
            "lower is better"
        )
    elif metric == "Roundtrip_RGB_BER":
        interpretation = (
            "lower is better"
        )
    else:
        interpretation = (
            "higher is better"
        )

    print(
        f"{metric:22s}: "
        f"{change:+.4f}% "
        f"({interpretation})"
    )


print("\n" + "=" * 80)
print("AI + CHAOTIC vs STANDARD EPVD")
print("=" * 80)


for metric in [
    "MSE",
    "PSNR",
    "SSIM",
    "Roundtrip_RGB_BER"
]:

    if metric not in overall_indexed.columns:
        continue

    change = improvement(
        chaotic_row[metric],
        standard_row[metric]
    )

    if metric == "MSE":
        interpretation = (
            "lower is better"
        )
    elif metric == "Roundtrip_RGB_BER":
        interpretation = (
            "lower is better"
        )
    else:
        interpretation = (
            "higher is better"
        )

    print(
        f"{metric:22s}: "
        f"{change:+.4f}% "
        f"({interpretation})"
    )


print("\n" + "=" * 80)
print("AI + CHAOTIC vs AI-EPVD")
print("=" * 80)


for metric in [
    "MSE",
    "PSNR",
    "SSIM",
    "Roundtrip_RGB_BER"
]:

    if metric not in overall_indexed.columns:
        continue

    change = improvement(
        chaotic_row[metric],
        ai_row[metric]
    )

    if metric == "MSE":
        interpretation = (
            "lower is better"
        )
    elif metric == "Roundtrip_RGB_BER":
        interpretation = (
            "lower is better"
        )
    else:
        interpretation = (
            "higher is better"
        )

    print(
        f"{metric:22s}: "
        f"{change:+.4f}% "
        f"({interpretation})"
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "final_three_way_bpp_comparison.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


overall_path = os.path.join(
    RESULTS_DIR,
    "final_three_way_overall.csv"
)

overall.to_csv(
    overall_path,
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"BPP comparison : {summary_path}"
)

print(
    f"Overall summary: {overall_path}"
)

print("\n" + "=" * 80)
print("FINAL COMPARISON COMPLETE")
print("=" * 80)