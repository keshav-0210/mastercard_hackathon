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
        # class_weight="balanced" is required here: at a realistic ~2% fraud rate, an unweighted
        # classifier with a naive 0.5 threshold learns to predict "legit" for any attack family
        # whose feature profile sits close to the legitimate distribution (e.g. trusted_device),
        # collapsing its recall to near zero regardless of how much training data it gets.
        self.pipeline = Pipeline([("features", transformer), ("classifier", HistGradientBoostingClassifier(random_state=42, class_weight="balanced"))])

    def fit(self, data: pd.DataFrame) -> None:
        self.pipeline.fit(data[FEATURES], data["is_fraud"])

    def predict(self, data: pd.DataFrame) -> pd.Series:
        probabilities = self.pipeline.predict_proba(data[FEATURES])[:, 1]
        return pd.Series((probabilities >= 0.5).astype(int), index=data.index)

    def _family_metrics(self, data: pd.DataFrame, probabilities: pd.Series) -> dict[str, dict[str, float | int]]:
        if "attack_family" not in data.columns:
            return {}

        family_metrics: dict[str, dict[str, float | int]] = {}
        for family, family_data in data[data["is_fraud"] == 1].groupby("attack_family", dropna=False):
            family_index = family_data.index
            family_probs = probabilities.loc[family_index]
            family_predictions = (family_probs >= 0.5).astype(int)
            family_labels = family_data["is_fraud"]
            family_metrics[family] = {
                "precision": float(precision_score(family_labels, family_predictions, zero_division=0)),
                "recall": float(recall_score(family_labels, family_predictions, zero_division=0)),
                "f1": float(f1_score(family_labels, family_predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(family_labels, family_probs)) if family_labels.nunique() > 1 else 0.5,
                "support": int(len(family_data)),
            }
        return family_metrics

    def evaluate(self, data: pd.DataFrame) -> dict:
        probabilities = pd.Series(self.pipeline.predict_proba(data[FEATURES])[:, 1], index=data.index)
        predictions = (probabilities >= 0.5).astype(int)
        labels = data["is_fraud"]
        result = {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if labels.nunique() > 1 else 0.5,
            "false_positive_rate": float(((predictions == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "by_attack_family": self._family_metrics(data, probabilities),
        }
        return result
