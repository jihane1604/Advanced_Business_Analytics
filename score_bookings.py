from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from preprocessing import DROP_ALWAYS, HotelBookingPreprocessor

MONTH_TO_NUM = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


class ModelInputTransformer(BaseEstimator, TransformerMixin):
    """
    Apply fitted categorical encoder, then align to the selected feature list.
    """

    def __init__(self, encoder, selected_features: List[str], target_col: str = "is_canceled"):
        self.encoder = encoder
        self.selected_features = selected_features
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_model = X.copy()
        if self.target_col in X_model.columns:
            X_model = X_model.drop(columns=[self.target_col])

        cat_cols = self._resolve_cat_columns(X_model)
        num_cols = [c for c in X_model.columns if c not in cat_cols]

        X_num = X_model[num_cols].copy()

        if cat_cols:
            X_cat_input = X_model.reindex(columns=cat_cols, fill_value="__MISSING__")
            X_cat = pd.DataFrame(
                self.encoder.transform(X_cat_input),
                columns=cat_cols,
                index=X_model.index,
            )
            X_encoded = pd.concat([X_num, X_cat], axis=1)
        else:
            X_encoded = X_num

        # Keep training feature order; fill absent features with 0.
        X_aligned = X_encoded.reindex(columns=self.selected_features, fill_value=0)
        return X_aligned

    def _resolve_cat_columns(self, X: pd.DataFrame) -> List[str]:
        # Preferred: exact training categorical columns captured by sklearn.
        if hasattr(self.encoder, "feature_names_in_"):
            return list(self.encoder.feature_names_in_)
        return X.select_dtypes(include=["object", "category"]).columns.tolist()


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (base_dir / p).resolve()


def _normalize_display_fields(
    df: pd.DataFrame, normalize_arrival_date: bool = True, normalize_company: bool = True
) -> pd.DataFrame:
    out = df.copy()

    if normalize_arrival_date and "arrival_date" in out.columns:
        # Use date components when available to avoid locale/ambiguity issues.
        if {
            "arrival_date_year",
            "arrival_date_month",
            "arrival_date_day_of_month",
        }.issubset(out.columns):
            month_num = (
                out["arrival_date_month"]
                .astype("string")
                .str.strip()
                .map(MONTH_TO_NUM)
            )
            date_from_parts = pd.to_datetime(
                dict(
                    year=pd.to_numeric(out["arrival_date_year"], errors="coerce"),
                    month=pd.to_numeric(month_num, errors="coerce"),
                    day=pd.to_numeric(out["arrival_date_day_of_month"], errors="coerce"),
                ),
                errors="coerce",
            )
            fallback = pd.to_datetime(out["arrival_date"], errors="coerce", dayfirst=True)
            parsed = date_from_parts.fillna(fallback)
            out["arrival_date"] = parsed.dt.strftime("%d/%m/%Y").fillna("")
        else:
            parsed = pd.to_datetime(out["arrival_date"], errors="coerce", dayfirst=True)
            out["arrival_date"] = parsed.dt.strftime("%d/%m/%Y").fillna("")

    if normalize_company and "company" in out.columns:
        out["company"] = (
            out["company"]
            .astype("string")
            .str.strip()
            .replace(["", "nan", "NaN", "<NA>", "NoneType"], pd.NA)
            .fillna("None")
            .astype(str)
        )

    return out


def _drop_artifact_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    tracking_cols = {
        "predicted_by_model",
        "is_canceled_pred",
        "pred_proba",
        "threshold_used",
        "feature_set_used",
        "company",
    }
    core_cols = [c for c in df.columns if c not in tracking_cols]
    if not core_cols:
        return df

    core_non_empty = (
        df[core_cols]
        .astype("string")
        .apply(lambda s: s.str.strip().replace("<NA>", ""))
        .ne("")
        .sum(axis=1)
    )
    only_empty_core = core_non_empty == 0

    company_none = pd.Series(True, index=df.index)
    if "company" in df.columns:
        company_none = df["company"].astype("string").str.strip().isin(
            ["None", "", "nan", "NaN", "<NA>"]
        )

    pred_false = pd.Series(True, index=df.index)
    if "predicted_by_model" in df.columns:
        pred_false = (
            df["predicted_by_model"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["false", "0", "", "nan", "<na>"])
        )

    mask_artifact = only_empty_core & company_none & pred_false
    if mask_artifact.any():
        return df.loc[~mask_artifact].copy()
    return df


def build_scoring_pipeline(
    model,
    encoder,
    selected_features: List[str],
    target_col: str,
    country_lookup_path: Optional[Path],
    drop_duplicates: bool,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                HotelBookingPreprocessor(
                    country_lookup=country_lookup_path,
                    drop_duplicates=drop_duplicates,
                    drop_model_cols=True,
                ),
            ),
            (
                "model_input",
                ModelInputTransformer(
                    encoder=encoder,
                    selected_features=selected_features,
                    target_col=target_col,
                ),
            ),
            ("model", model),
        ]
    )


def print_validation_report(output: pd.DataFrame, expected_threshold: float, expected_feature_set: str) -> None:
    required_cols = [
        "pred_proba",
        "is_canceled_pred",
        "predicted_by_model",
        "threshold_used",
        "feature_set_used",
    ]
    missing_cols = [c for c in required_cols if c not in output.columns]

    proba = output["pred_proba"] if "pred_proba" in output.columns else pd.Series(dtype=float)
    labels = (
        output["is_canceled_pred"]
        if "is_canceled_pred" in output.columns
        else pd.Series(dtype=float)
    )

    proba_in_range = bool(((proba >= 0) & (proba <= 1)).all()) if not proba.empty else False
    proba_nan_n = int(proba.isna().sum()) if not proba.empty else 0
    label_values = sorted(pd.unique(labels)) if not labels.empty else []
    labels_binary = set(label_values).issubset({0, 1}) if label_values else False

    if "pred_proba" in output.columns and "is_canceled_pred" in output.columns:
        expected_labels = (output["pred_proba"] >= expected_threshold).astype(int)
        label_threshold_match = bool((expected_labels == output["is_canceled_pred"]).all())
    else:
        label_threshold_match = False

    threshold_unique = sorted(output["threshold_used"].unique().tolist()) if "threshold_used" in output.columns else []
    feature_set_unique = sorted(output["feature_set_used"].astype(str).unique().tolist()) if "feature_set_used" in output.columns else []

    print("\nValidation report")
    print(f"Required columns present: {len(missing_cols) == 0}")
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    print(f"pred_proba in [0,1]: {proba_in_range}")
    print(f"pred_proba NaN count: {proba_nan_n}")
    print(f"is_canceled_pred binary (0/1): {labels_binary} | values: {label_values}")
    print(f"is_canceled_pred matches threshold rule: {label_threshold_match}")
    print(f"threshold_used unique: {threshold_unique} | expected: {expected_threshold}")
    print(f"feature_set_used unique: {feature_set_unique} | expected: {expected_feature_set}")

    if not output.empty and "is_canceled_pred" in output.columns:
        cancel_rate = float(output["is_canceled_pred"].mean())
        print(f"Predicted cancellation rate: {cancel_rate:.4f}")


def _ensure_tracking_cols(df: pd.DataFrame, is_historical: bool) -> pd.DataFrame:
    out = df.copy()
    if "predicted_by_model" not in out.columns:
        out["predicted_by_model"] = False if is_historical else True
    elif is_historical:
        out["predicted_by_model"] = out["predicted_by_model"].fillna(False)
    else:
        out["predicted_by_model"] = True

    for col in ["is_canceled_pred", "pred_proba", "threshold_used", "feature_set_used"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def append_to_historical(historical_csv: Path, new_rows: pd.DataFrame) -> tuple[int, int]:
    if historical_csv.exists():
        historical = pd.read_csv(historical_csv)
    else:
        historical = pd.DataFrame()

    # Normalize historical display fields too, so arrival_date stays consistent dd/mm/yyyy.
    historical = _normalize_display_fields(
        historical, normalize_arrival_date=True, normalize_company=True
    )
    historical = _drop_artifact_rows(historical)
    new_rows = _normalize_display_fields(new_rows)

    historical = _ensure_tracking_cols(historical, is_historical=True)
    new_rows = _ensure_tracking_cols(new_rows, is_historical=False)

    all_cols = sorted(set(historical.columns).union(new_rows.columns))
    historical_aligned = historical.reindex(columns=all_cols)
    new_rows_aligned = new_rows.reindex(columns=all_cols)

    combined = pd.concat([historical_aligned, new_rows_aligned], ignore_index=True)
    historical_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(historical_csv, index=False)
    return len(historical), len(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score hotel bookings with the trained model pipeline."
    )
    parser.add_argument("--input-csv", required=True, help="Path to raw input CSV.")
    parser.add_argument(
        "--output-csv",
        default="data/scoring/scored_bookings.csv",
        help="Path to output CSV with predictions.",
    )
    parser.add_argument(
        "--model-path",
        default="models/final/final_model.pkl",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--meta-path",
        default="models/final/final_model_meta.json",
        help="Path to best model metadata JSON.",
    )
    parser.add_argument(
        "--encoder-path",
        default="models/ordinal_encoder.pkl",
        help="Path to fitted ordinal encoder.",
    )
    parser.add_argument(
        "--feature-sets-path",
        default="data/feature_sets/feature_sets.json",
        help="Path to feature sets JSON.",
    )
    parser.add_argument(
        "--country-lookup-path",
        default="data/country_lookup.csv",
        help="Path to country lookup CSV.",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Feature set key override (default: use metadata feature_set).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold override (default: use metadata threshold).",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Apply deduplication during preprocessing (default: keep all rows).",
    )
    parser.add_argument(
        "--validation-report",
        action="store_true",
        help="Print post-scoring data quality checks for predictions.",
    )
    parser.add_argument(
        "--historical-csv",
        default="data/dashboard_data/historical_data.csv",
        help="Historical dashboard dataset to append newly scored rows to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    input_csv = _resolve_path(args.input_csv, base_dir)
    output_csv = _resolve_path(args.output_csv, base_dir)
    model_path = _resolve_path(args.model_path, base_dir)
    meta_path = _resolve_path(args.meta_path, base_dir)
    encoder_path = _resolve_path(args.encoder_path, base_dir)
    feature_sets_path = _resolve_path(args.feature_sets_path, base_dir)
    country_lookup_path = _resolve_path(args.country_lookup_path, base_dir)
    historical_csv = _resolve_path(args.historical_csv, base_dir)

    df_input = pd.read_csv(input_csv)
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(feature_sets_path, "r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    target_col = meta.get("target", "is_canceled")
    feature_set_key = args.feature_set or meta.get("feature_set", "all_train_cols")
    meta_threshold = meta.get("threshold")
    if meta_threshold is None and isinstance(meta.get("best_model_meta"), dict):
        meta_threshold = meta["best_model_meta"].get("threshold")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(meta_threshold if meta_threshold is not None else 0.5)
    )

    if feature_set_key not in feature_sets:
        available = ", ".join(feature_sets.keys())
        raise ValueError(
            f"Feature set '{feature_set_key}' not found. Available: {available}"
        )

    selected_features = feature_sets[feature_set_key]

    preprocessor = HotelBookingPreprocessor(
        country_lookup=country_lookup_path,
        drop_duplicates=args.drop_duplicates,
        drop_model_cols=False,
    )
    preprocessor.fit(df_input)
    live_engineered = preprocessor.transform(df_input)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    live_model_frame = live_engineered.drop(
        columns=[c for c in DROP_ALWAYS if c in live_engineered.columns]
    )
    model_input = ModelInputTransformer(
        encoder=encoder,
        selected_features=selected_features,
        target_col=target_col,
    )
    X_scored = model_input.transform(live_model_frame)
    proba = model.predict_proba(X_scored)[:, 1]
    pred = (proba >= threshold).astype(int)

    output = _normalize_display_fields(live_engineered.copy())
    output["is_canceled_pred"] = pred
    output["pred_proba"] = proba
    output["predicted_by_model"] = True
    output["threshold_used"] = threshold
    output["feature_set_used"] = feature_set_key

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    old_hist_n, new_hist_n = append_to_historical(historical_csv, output)

    dropped_rows = len(df_input) - len(live_engineered)
    print(f"Rows scored: {len(output):,}")
    if dropped_rows > 0:
        print(f"Rows dropped during preprocessing: {dropped_rows:,}")
    print(f"Output saved: {output_csv}")
    print(
        f"Historical updated: {historical_csv} "
        f"(rows: {old_hist_n:,} -> {new_hist_n:,})"
    )
    print(f"Feature set: {feature_set_key} | Threshold: {threshold:.4f}")
    if args.validation_report:
        print_validation_report(output, threshold, feature_set_key)


if __name__ == "__main__":
    main()
