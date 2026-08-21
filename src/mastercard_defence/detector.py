from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel"]


class FraudDetector:
    def __init__(self) -> None:
        transformer = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), ["channel"])],
            remainder="passthrough",
        )
        self.pipeline = Pipeline([( "features", transformer), ("classifier", HistGradientBoostingClassifier(random_state=42))])

    def fit(self, data: pd.DataFrame) -> None:
        self.pipeline.fit(data[FEATURES], data["is_fraud"])

    def evaluate(self, data: pd.DataFrame) -> dict:
        probabilities = self.pipeline.predict_proba(data[FEATURES])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        labels = data["is_fraud"]
        return {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if labels.nunique() > 1 else 0.5,
            "false_positive_rate": float(((predictions == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        }
