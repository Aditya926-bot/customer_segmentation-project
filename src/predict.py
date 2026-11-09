
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


# Project root and saved model path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "artifacts.pkl"


def load_artifacts():
    """Load the saved model and metadata."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {ARTIFACT_PATH}"
        )

    # Only load pickle files created by you and trusted.
    with ARTIFACT_PATH.open("rb") as file:
        artifacts = pickle.load(file)

    return artifacts


# Load once when this module is imported.
ARTIFACTS = load_artifacts()


def scale_model_features(customer_features):
    """
    Apply the saved scaling parameters to the five model features.
    This reproduces the transformation used during model training.
    """
    feature_names = ARTIFACTS["features"]
    scaling = ARTIFACTS["scaling"]

    missing = [
        name for name in feature_names
        if name not in customer_features.columns
    ]
    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    scaled_columns = []

    for name in feature_names:
        params = scaling[name]

        values = pd.to_numeric(
            customer_features[name], errors="raise"
        ).astype(float).to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Feature '{name}' contains invalid numeric values."
            )

        # Apply log1p only to features trained with the log transform.
        if params["log"]:
            if np.any(values < 0):
                raise ValueError(
                    f"Feature '{name}' cannot be negative."
                )
            values = np.log1p(values)

        scale = float(params["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Invalid saved scaling value for '{name}'."
            )

        scaled = (
            values - float(params["mean"])
        ) / scale

        scaled_columns.append(scaled)

    return np.column_stack(scaled_columns)


def predict_customer(customer_features):
    """
    Predict a customer's cluster and return the segment,
    feature-distance explanation, and profile comparisons.
    """
    if len(customer_features) != 1:
        raise ValueError(
            "Prediction requires exactly one customer profile."
        )

    model = ARTIFACTS["model"]
    feature_names = ARTIFACTS["features"]
    segment_names = ARTIFACTS["segment_names"]

    scaled = scale_model_features(customer_features)

    # Predict the nearest KMeans cluster.
    cluster_id = int(model.predict(scaled)[0])

    # Calculate distance to every cluster center.
    distances = model.transform(scaled)[0]
    center_ids = list(range(len(distances)))

    ranked = sorted(
        center_ids,
        key=lambda idx: float(distances[idx])
    )

    nearest_distance = float(distances[cluster_id])
    runner_up_id = next(
        (idx for idx in ranked if idx != cluster_id),
        None
    )

    # Explain which features differ from the assigned center.
    center = model.cluster_centers_[cluster_id]
    differences = scaled[0] - center

    feature_reasons = []
    for index, name in enumerate(feature_names):
        feature_reasons.append({
            "feature": name,
            "scaled_customer_value": float(scaled[0, index]),
            "cluster_center_value": float(center[index]),
            "difference": float(differences[index]),
            "absolute_difference": float(abs(differences[index])),
        })

    feature_reasons.sort(
        key=lambda item: item["absolute_difference"],
        reverse=True
    )

    # Compare other customer profile features with cluster medians.
    profile_reasons = []

    cluster_medians = ARTIFACTS.get("cluster_medians")
    overall_iqr = ARTIFACTS.get("overall_iqr")

    if cluster_medians is not None and cluster_id in cluster_medians.index:
        medians = cluster_medians.loc[cluster_id]

        for feature in ARTIFACTS.get("profile_features", []):
            if feature not in customer_features.columns:
                continue
            if feature not in medians.index:
                continue

            value = pd.to_numeric(
                customer_features[feature], errors="coerce"
            ).iloc[0]
            median = medians[feature]

            if pd.isna(value) or pd.isna(median):
                continue

            value = float(value)
            median = float(median)

            # Use overall IQR to avoid treating tiny differences
            # as meaningful. If unavailable, use a small tolerance.
            tolerance = 0.0
            if overall_iqr is not None and feature in overall_iqr.index:
                iqr_value = overall_iqr[feature]
                if pd.notna(iqr_value):
                    tolerance = 0.05 * abs(float(iqr_value))

            if value > median + tolerance:
                comparison = "Higher"
            elif value < median - tolerance:
                comparison = "Lower"
            else:
                comparison = "Similar"

            profile_reasons.append({
                "feature": feature,
                "customer_value": value,
                "cluster_median": median,
                "comparison": comparison,
            })

    return {
        "customer_id": str(
            customer_features["Customer ID"].iloc[0]
        ) if "Customer ID" in customer_features.columns else "Unknown",
        "cluster_id": cluster_id,
        "segment_name": segment_names.get(
            cluster_id, f"Cluster {cluster_id}"
        ),
        "distance_to_cluster_center": nearest_distance,
        "runner_up_cluster": runner_up_id,
        "runner_up_distance": (
            float(distances[runner_up_id])
            if runner_up_id is not None else None
        ),
        "feature_reasons": feature_reasons,
        "profile_reasons": profile_reasons,
        "all_cluster_distances": {
            int(idx): float(distance)
            for idx, distance in enumerate(distances)
        },
    }
