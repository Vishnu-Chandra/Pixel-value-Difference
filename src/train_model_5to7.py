import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "labeled_features_5to7.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest_5to7.pkl"
)

FEATURE_COLUMNS = [
    "variance",
    "entropy",
    "edge_density",
    "gradient_magnitude",
    "mean_pixel_difference",
    "max_pixel_difference",
    "std_pixel_difference"
]

print("=" * 70)
print("Training Random Forest: 5–7 Bit Adaptive EPVD")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

print("\nEmbedding-bit distribution:")
print(
    df["embedding_bits"]
    .value_counts()
    .sort_index()
)

X = df[FEATURE_COLUMNS]
y = df["embedding_bits"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[5, 6, 7],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[5, 6, 7]
)

print("Confusion Matrix:")
print("Rows = Actual")
print("Columns = Predicted")
print("\nClasses: [5, 6, 7]")
print(cm)

importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(
    importance.to_string(index=False)
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)

print("=" * 70)