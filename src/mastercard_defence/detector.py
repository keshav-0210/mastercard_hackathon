from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel"]
OPTIONAL_RELATIONSHIP_FEATURES = ["account_id", "device_id", "beneficiary_id"]


def _select_features(data: pd.DataFrame) -> list[str]:
    return FEATURES + [column for column in OPTIONAL_RELATIONSHIP_FEATURES if column in data.columns]


class RelationshipFeaturizer(BaseEstimator, TransformerMixin):
    """Derives per-entity history and shared-device/shared-beneficiary counts from fit-time-frozen
    lookup tables (like a scaler fit on train). Requires the caller's entities to persist across
    fit/transform calls (see EntityRegistry) so lookups reflect real repeat behaviour rather than an
    artifact of each batch having its own private ID pool. A no-op when the id columns are absent."""

    def fit(self, X: pd.DataFrame, y=None):
        self.global_mean_amount_ = float(X["amount"].mean()) if "amount" in X.columns else 0.0
        self.account_counts_ = X["account_id"].value_counts().to_dict() if "account_id" in X.columns else {}
        self.account_mean_amount_ = (
            X.groupby("account_id")["amount"].mean().to_dict()
            if "account_id" in X.columns and "amount" in X.columns
            else {}
        )
        self.device_account_counts_ = (
            X.groupby("device_id")["account_id"].nunique().to_dict()
            if "device_id" in X.columns and "account_id" in X.columns
            else {}
        )
        self.beneficiary_account_counts_ = (
            X.groupby("beneficiary_id")["account_id"].nunique().to_dict()
            if "beneficiary_id" in X.columns and "account_id" in X.columns
            else {}
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        engineered = X.copy()
        if "account_id" in engineered.columns:
            engineered["account_prior_count"] = engineered["account_id"].map(self.account_counts_).fillna(0)
            engineered["account_prior_mean_amount"] = engineered["account_id"].map(self.account_mean_amount_).fillna(self.global_mean_amount_)
        if "device_id" in engineered.columns:
            engineered["device_shared_account_count"] = engineered["device_id"].map(self.device_account_counts_).fillna(1)
        if "beneficiary_id" in engineered.columns:
            engineered["beneficiary_shared_account_count"] = engineered["beneficiary_id"].map(self.beneficiary_account_counts_).fillna(1)
        return engineered.drop(columns=[column for column in OPTIONAL_RELATIONSHIP_FEATURES if column in engineered.columns])


class FraudDetector:
    def __init__(self, target_fpr_ceiling: float = 0.02) -> None:
        transformer = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), ["channel"])],
            remainder="passthrough",
        )
        self.pipeline = Pipeline([
            ("relationship", RelationshipFeaturizer()),
            ("features", transformer),
            ("classifier", HistGradientBoostingClassifier(random_state=42, class_weight="balanced")),
        ])
        self.target_fpr_ceiling = target_fpr_ceiling
        self.threshold = 0.5

    def fit(self, data: pd.DataFrame, calibration_data: pd.DataFrame | None = None) -> None:
        self.pipeline.fit(data[_select_features(data)], data["is_fraud"])
        self._calibrate_threshold(calibration_data if calibration_data is not None else data[data["is_fraud"] == 0])

    def _calibrate_threshold(self, data: pd.DataFrame) -> None:
        probabilities = self.pipeline.predict_proba(data[_select_features(data)])[:, 1]
        if probabilities.size == 0:
            self.threshold = 0.5
            return
        boundary = float(np.quantile(probabilities, 1.0 - self.target_fpr_ceiling, method="higher"))
        self.threshold = float(np.nextafter(boundary, np.inf))

    def predict(self, data: pd.DataFrame) -> pd.Series:
        probabilities = self.pipeline.predict_proba(data[_select_features(data)])[:, 1]
        return pd.Series((probabilities >= self.threshold).astype(int), index=data.index)

    def _family_metrics(self, data: pd.DataFrame, probabilities: pd.Series) -> dict[str, dict[str, float | int]]:
        if "attack_family" not in data.columns:
            return {}

        family_metrics: dict[str, dict[str, float | int]] = {}
        for family, family_data in data[data["is_fraud"] == 1].groupby("attack_family", dropna=False):
            family_index = family_data.index
            family_probs = probabilities.loc[family_index]
            family_predictions = (family_probs >= self.threshold).astype(int)
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
        probabilities = pd.Series(self.pipeline.predict_proba(data[_select_features(data)])[:, 1], index=data.index)
        predictions = (probabilities >= self.threshold).astype(int)
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
