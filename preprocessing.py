from __future__ import annotations

# preprocessing logic for eda and feature selection
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Fixed winsorization thresholds from EDA training run.
CAPS = {
    "adr": 285.0,
    "lead_time": 347.0,
    "stays_in_weekend_nights": 4.0,
    "stays_in_week_nights": 10.0,
    "booking_changes": 4.0,
    "days_in_waiting_list": 58.0,
    "previous_cancellations": 2.0,
    "previous_bookings_not_canceled": 9.0,
}

# Business-rule caps used in EDA.
LIMITS = {
    "adults": 6,
    "children": 3,
    "babies": 2,
}

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

# Columns dropped in feature_selection/model_development before encoding.
DROP_ALWAYS = [
    "reservation_status",       # leakage
    "reservation_status_date",  # leakage
    "arrival_date",             # redundant with year/month/day
    "arrival_weekday",          # redundant with is_weekend_arrival
    "lead_time_bucket",         # categorical bin of lead_time, keep original
    "country",                  # high cardinality, continent used instead
    "country_code_iso3",        # redundant with continent
    "agent",                    # ID column, high cardinality
    "company",                  # ID column, high cardinality
]


def _normalize_country_lookup(
    country_lookup: Union[pd.DataFrame, str, Path]
) -> pd.DataFrame:
    if isinstance(country_lookup, (str, Path)):
        country_lookup = pd.read_csv(country_lookup)
    if "country_code" in country_lookup.columns:
        return country_lookup.rename(columns={"country_code": "country"}).copy()
    return country_lookup.copy()


def preprocess_new_data(
    df_raw: pd.DataFrame,
    country_lookup: Optional[Union[pd.DataFrame, str, Path]] = None,
    drop_duplicates: bool = True,
    drop_model_cols: bool = False,
) -> pd.DataFrame:
    """Apply the same preprocessing logic used in EDA to new/raw data."""
    df_clean = df_raw.copy()

    # Base cleaning.
    df_clean["children"] = df_clean["children"].fillna(0).astype(int)
    df_clean["country"] = df_clean["country"].fillna("Unknown")

    df_clean["agent"] = (
        df_clean["agent"]
        .replace(["NULL", "null", "Null"], np.nan)
        .fillna("Direct")
        .astype(str)
    )
    df_clean["company"] = (
        df_clean["company"]
        .replace(["NULL", "null", "Null"], np.nan)
        .fillna("None")
        .astype(str)
    )

    # Remove phantom bookings.
    phantom_mask = (
        (df_clean["adults"] == 0)
        & (df_clean["children"] == 0)
        & (df_clean["babies"] == 0)
    )
    df_clean = df_clean.loc[~phantom_mask].copy()

    # Remove negative ADR.
    neg_mask = df_clean["adr"] < 0
    adr_mean = df_clean.loc[~neg_mask, "adr"].mean()
    df_clean.loc[neg_mask, "adr"] = adr_mean

    # Winsorization + flags/logs.
    df_clean["adr_was_capped"] = (df_clean["adr"] > CAPS["adr"]).astype(int)
    df_clean["adr"] = df_clean["adr"].clip(upper=CAPS["adr"])

    df_clean["lead_time_was_capped"] = (
        df_clean["lead_time"] > CAPS["lead_time"]
    ).astype(int)
    df_clean["lead_time"] = df_clean["lead_time"].clip(upper=CAPS["lead_time"])
    df_clean["lead_time_log"] = np.log1p(df_clean["lead_time"])

    df_clean["stays_weekend_was_capped"] = (
        df_clean["stays_in_weekend_nights"] > CAPS["stays_in_weekend_nights"]
    ).astype(int)
    df_clean["stays_in_weekend_nights"] = df_clean["stays_in_weekend_nights"].clip(
        upper=CAPS["stays_in_weekend_nights"]
    )

    df_clean["stays_week_was_capped"] = (
        df_clean["stays_in_week_nights"] > CAPS["stays_in_week_nights"]
    ).astype(int)
    df_clean["stays_in_week_nights"] = df_clean["stays_in_week_nights"].clip(
        upper=CAPS["stays_in_week_nights"]
    )

    df_clean["adults_was_capped"] = (df_clean["adults"] > LIMITS["adults"]).astype(int)
    df_clean["adults"] = df_clean["adults"].clip(upper=LIMITS["adults"])

    df_clean["has_children"] = (df_clean["children"] > 0).astype(int)
    df_clean["children_was_capped"] = (
        df_clean["children"] > LIMITS["children"]
    ).astype(int)
    df_clean["children"] = df_clean["children"].clip(upper=LIMITS["children"])

    df_clean["has_babies"] = (df_clean["babies"] > 0).astype(int)
    df_clean["babies_was_capped"] = (df_clean["babies"] > LIMITS["babies"]).astype(int)
    df_clean["babies"] = df_clean["babies"].clip(upper=LIMITS["babies"])

    df_clean["has_booking_changes"] = (df_clean["booking_changes"] > 0).astype(int)
    df_clean["booking_changes_was_capped"] = (
        df_clean["booking_changes"] > CAPS["booking_changes"]
    ).astype(int)
    df_clean["booking_changes"] = df_clean["booking_changes"].clip(
        upper=CAPS["booking_changes"]
    )

    df_clean["has_waiting_list"] = (df_clean["days_in_waiting_list"] > 0).astype(int)
    df_clean["days_waiting_was_capped"] = (
        df_clean["days_in_waiting_list"] > CAPS["days_in_waiting_list"]
    ).astype(int)
    df_clean["days_in_waiting_list"] = df_clean["days_in_waiting_list"].clip(
        upper=CAPS["days_in_waiting_list"]
    )
    df_clean["days_in_waiting_list_log"] = np.log1p(df_clean["days_in_waiting_list"])

    df_clean["previous_cancellations_raw"] = df_clean["previous_cancellations"]
    df_clean["previous_bookings_not_canceled_raw"] = df_clean[
        "previous_bookings_not_canceled"
    ]

    df_clean["previous_cancellations_was_capped"] = (
        df_clean["previous_cancellations_raw"] > CAPS["previous_cancellations"]
    ).astype(int)
    df_clean["previous_cancellations"] = df_clean["previous_cancellations_raw"].clip(
        upper=CAPS["previous_cancellations"]
    )
    df_clean["has_prev_cancellations"] = (df_clean["previous_cancellations"] > 0).astype(
        int
    )
    df_clean["previous_cancellations_log"] = np.log1p(df_clean["previous_cancellations"])

    df_clean["previous_bookings_not_canceled_was_capped"] = (
        df_clean["previous_bookings_not_canceled_raw"]
        > CAPS["previous_bookings_not_canceled"]
    ).astype(int)
    df_clean["previous_bookings_not_canceled"] = df_clean[
        "previous_bookings_not_canceled_raw"
    ].clip(upper=CAPS["previous_bookings_not_canceled"])
    df_clean["has_prev_not_canceled"] = (
        df_clean["previous_bookings_not_canceled"] > 0
    ).astype(int)
    df_clean["previous_bookings_not_canceled_log"] = np.log1p(
        df_clean["previous_bookings_not_canceled"]
    )

    df_clean["has_any_history"] = (
        (
            df_clean["previous_cancellations"]
            + df_clean["previous_bookings_not_canceled"]
        )
        > 0
    ).astype(int)
    prev_total = (
        df_clean["previous_cancellations"] + df_clean["previous_bookings_not_canceled"]
    )
    df_clean["prev_cancel_ratio"] = np.where(
        prev_total > 0, df_clean["previous_cancellations"] / prev_total, 0.0
    )

    # Feature engineering.
    df_clean["arrival_month_num"] = df_clean["arrival_date_month"].map(MONTH_TO_NUM)
    df_clean["total_nights"] = (
        df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]
    )
    df_clean["total_guests"] = df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    df_clean["is_room_changed"] = (
        df_clean["reserved_room_type"] != df_clean["assigned_room_type"]
    ).astype(int)

    df_clean["arrival_date"] = pd.to_datetime(
        df_clean["arrival_date_year"].astype(str)
        + "-"
        + df_clean["arrival_month_num"].astype("Int64").astype(str).str.zfill(2)
        + "-"
        + df_clean["arrival_date_day_of_month"].astype(str).str.zfill(2),
        errors="coerce",
    )
    df_clean["revenue_at_risk"] = (df_clean["adr"] * df_clean["total_nights"]).round(2)

    if country_lookup is not None:
        lookup = _normalize_country_lookup(country_lookup)
        df_clean = df_clean.merge(lookup, on="country", how="left")
        if "continent" in df_clean.columns:
            df_clean["continent"] = df_clean["continent"].fillna("Other")
        if "region" in df_clean.columns:
            df_clean["region"] = df_clean["region"].fillna("Other")

    df_clean["lead_time_bucket"] = pd.cut(
        df_clean["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, 9999],
        labels=[
            "Same week",
            "1-4 weeks",
            "1-3 months",
            "3-6 months",
            "6-12 months",
            "12+ months",
        ],
    )

    # Same EDA behavior: deduplicate at end.
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Modeling behavior from feature_selection/model_development.
    if drop_model_cols:
        cols_to_drop = [c for c in DROP_ALWAYS if c in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_to_drop)

    return df_clean


@dataclass
class HotelBookingPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper around preprocess_new_data()."""

    country_lookup: Optional[Union[pd.DataFrame, str, Path]] = None
    drop_duplicates: bool = True
    drop_model_cols: bool = False

    def fit(self, X, y=None):
        if isinstance(self.country_lookup, (str, Path)):
            self.country_lookup_ = pd.read_csv(self.country_lookup)
        else:
            self.country_lookup_ = self.country_lookup
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        lookup = getattr(self, "country_lookup_", self.country_lookup)
        return preprocess_new_data(
            X,
            country_lookup=lookup,
            drop_duplicates=self.drop_duplicates,
            drop_model_cols=self.drop_model_cols,
        )

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X):
        return X[self.columns_].copy()
