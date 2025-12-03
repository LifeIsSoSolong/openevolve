from typing import Any
import pandas as pd
import numpy as np
from mle_agent.metrics.base import CompetitionMetrics, InvalidSubmissionError

class BrazilSoybeanYieldMetrics(CompetitionMetrics):
    """
    Metric class for Brazilian Soybean Yield Prediction using MAPE for evaluation.
    Only months strictly BEFORE (cutoff_year, cutoff_month) are used.
    Default cutoff is (2025, 10) => exclude 2025-10 and any later month.
    """

    def __init__(self, value: str = "mape", higher_is_better: bool = False,
                 cutoff_year: int = 2025, cutoff_month: int = 10):
        # Lower MAPE is better.
        super().__init__(higher_is_better)
        self.value = value
        self.cutoff_year = cutoff_year
        self.cutoff_month = cutoff_month

    # ---------- helpers ----------

    def _require_cols(self, df: pd.DataFrame, name: str, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    def _require_cols_submission(self, df: pd.DataFrame, name: str, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise InvalidSubmissionError(f"{name} is missing required columns: {missing}")

    def _filter_before_cutoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with (year, month) strictly before (cutoff_year, cutoff_month)."""
        y = self.cutoff_year
        m = self.cutoff_month
        return df[(df["year"] < y) | ((df["year"] == y) & (df["month"] < m))]

    # ---------- metrics.evaluate ----------

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """Evaluate using MAPE. Extra rows in submission are ignored automatically."""
        required_cols = ['year', 'month', 'state', 'yield']

        # Basic validation
        if y_true.empty or y_pred.empty:
            raise ValueError("Input DataFrames (y_true or y_pred) cannot be empty.")
        self._require_cols(y_true, 'y_true', required_cols)
        self._require_cols(y_pred, 'y_pred', required_cols)

        # Apply cutoff filter (exclude 2025-10 and later by default)
        y_true_f = self._filter_before_cutoff(y_true)
        y_pred_f = self._filter_before_cutoff(y_pred)

        if y_true_f.empty:
            raise ValueError("After cutoff filtering, ground truth is empty.")
        if y_pred_f.empty:
            raise ValueError("After cutoff filtering, submission is empty.")

        # Merge for alignment
        merged = pd.merge(
            y_true_f,
            y_pred_f,
            on=['year', 'month', 'state'],
            suffixes=('_true', '_pred'),
            how='inner'
        )
        if merged.empty:
            raise ValueError("No overlapping keys between submission and ground truth after cutoff filtering.")

        # Drop rows where ground-truth yield is missing
        merged = merged.dropna(subset=['yield_true'])
        if merged.empty:
            raise ValueError("All ground-truth yield values are missing after cutoff; cannot compute MAPE.")

        y_true_vals = merged['yield_true'].values
        y_pred_vals = merged['yield_pred'].values

        mask = np.abs(y_true_vals) > 1e-12
        if not mask.any():
            raise ValueError("Ground truth contains only zeros after cutoff; cannot compute MAPE.")

        mape = np.mean(np.abs((y_true_vals[mask] - y_pred_vals[mask]) / y_true_vals[mask])) * 100
        return mape

    # ---------- metrics.validate_submission ----------

    def validate_submission(self, submission: pd.DataFrame, ground_truth: pd.DataFrame):
        """
        Validate submission structure and keys against ground truth using only
        rows strictly before (cutoff_year, cutoff_month).
        Requires exact match of (year, month, state) keys after filtering and
        dropping NaN yields from the ground truth (and submission).
        """

        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame.")

        if submission.empty:
            raise InvalidSubmissionError("Submission DataFrame cannot be empty.")
        if ground_truth.empty:
            raise InvalidSubmissionError("Ground truth DataFrame cannot be empty.")

        required_cols = ['year', 'month', 'state', 'yield']
        self._require_cols_submission(submission, 'submission', required_cols)
        self._require_cols_submission(ground_truth, 'ground_truth', required_cols)

        # Apply cutoff filter (exclude 2025-10 and later by default)
        gt_f = self._filter_before_cutoff(ground_truth).copy()
        sub_f = self._filter_before_cutoff(submission).copy()

        if gt_f.empty:
            raise InvalidSubmissionError("Ground truth has no rows before cutoff; cannot validate.")
        if sub_f.empty:
            raise InvalidSubmissionError("Submission has no rows before cutoff; cannot validate.")

        gt_f = gt_f.dropna(subset=['yield'])
        if gt_f.empty:
            raise InvalidSubmissionError("Ground truth has no valid yield values before cutoff for evaluation.")

        # Submission rows with NaN yield are ignored for scoring but kept for key coverage
        sub_f = sub_f.copy()

        extra_cols = [col for col in sub_f.columns if col not in required_cols]
        if extra_cols:
            raise InvalidSubmissionError(f"Submission contains unexpected columns: {extra_cols}")

        if sub_f.duplicated(subset=['year', 'month', 'state']).any():
            dup_keys = sub_f[sub_f.duplicated(subset=['year','month','state'], keep=False)][['year','month','state']].drop_duplicates().values[:5]
            raise InvalidSubmissionError(f"Submission contains duplicate keys: {dup_keys}")

        gt_keys = set(map(tuple, gt_f[['year','month','state']].values))
        sub_keys = set(map(tuple, sub_f[['year','month','state']].values))

        missing_in_sub = gt_keys - sub_keys
        if missing_in_sub:
            raise InvalidSubmissionError(
                "Submission is missing some required keys (before cutoff). "
                f"Examples: {list(missing_in_sub)[:5]}"
            )

        return "Submission is valid."
