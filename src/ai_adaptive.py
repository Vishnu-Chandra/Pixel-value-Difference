import os
import joblib
import pandas as pd

from features import extract_features


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest_5to7.pkl"
)


# ============================================================
# Feature Order
# ============================================================

FEATURE_COLUMNS = [
    "variance",
    "entropy",
    "edge_density",
    "gradient_magnitude",
    "mean_pixel_difference",
    "max_pixel_difference",
    "std_pixel_difference"
]


# ============================================================
# Adaptive Bit Predictor
# ============================================================

class AdaptiveBitPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Random Forest model not found: {model_path}"
            )

        self.model = joblib.load(
            model_path
        )


    # ========================================================
    # Predict One Block
    # ========================================================

    def predict_block(
        self,
        block
    ):

        features = extract_features(
            block
        )

        feature_df = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS
        )

        prediction = self.model.predict(
            feature_df
        )[0]

        return int(prediction)


    # ========================================================
    # Predict Multiple Blocks
    # ========================================================

    def predict_blocks(
        self,
        blocks
    ):

        feature_rows = [
            extract_features(block)
            for block in blocks
        ]

        feature_df = pd.DataFrame(
            feature_rows,
            columns=FEATURE_COLUMNS
        )

        predictions = self.model.predict(
            feature_df
        )

        return predictions.astype(int)