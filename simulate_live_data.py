from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate live booking data by spreading arrival dates forward in time."
    )
    parser.add_argument(
        "--input-csv",
        default="data/hotel_bookings_live_sample_300.csv",
        help="Input CSV to simulate from.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/simulations",
        help="Directory to save simulated CSV runs.",
    )
    parser.add_argument(
        "--simulation-name",
        default="live_arrival_shift",
        help="Simulation name used in output filename.",
    )
    parser.add_argument(
        "--start-date",
        default="01/09/2017",
        help="Start date (DD/MM/YYYY) for simulated arrivals.",
    )
    parser.add_argument(
        "--min-step-days",
        type=int,
        default=1,
        help="Minimum day increment between consecutive rows.",
    )
    parser.add_argument(
        "--max-step-days",
        type=int,
        default=3,
        help="Maximum day increment between consecutive rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible spacing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    input_csv = (base_dir / args.input_csv).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_step_days < 1 or args.max_step_days < args.min_step_days:
        raise ValueError("Invalid day step range. Ensure 1 <= min-step-days <= max-step-days.")

    df = pd.read_csv(input_csv).copy()
    n = len(df)
    if n == 0:
        raise ValueError("Input CSV has no rows.")

    start_dt = datetime.strptime(args.start_date, "%d/%m/%Y").date()
    rng = np.random.default_rng(args.seed)
    steps = rng.integers(args.min_step_days, args.max_step_days + 1, size=n)

    arrival_dates = []
    current_date = start_dt
    for i in range(n):
        if i > 0:
            current_date = current_date + timedelta(days=int(steps[i]))
        arrival_dates.append(current_date)

    dt_series = pd.to_datetime(pd.Series(arrival_dates))
    iso = dt_series.dt.isocalendar()

    df["arrival_date_year"] = dt_series.dt.year.astype(int)
    df["arrival_date_month"] = dt_series.dt.month_name()
    df["arrival_date_day_of_month"] = dt_series.dt.day.astype(int)
    df["arrival_date_week_number"] = iso.week.astype(int)

    if "arrival_date" in df.columns:
        df["arrival_date"] = dt_series.dt.strftime("%d/%m/%Y")
    if "arrival_weekday" in df.columns:
        df["arrival_weekday"] = dt_series.dt.day_name()
    if "is_weekend_arrival" in df.columns:
        df["is_weekend_arrival"] = (dt_series.dt.weekday >= 5).astype(int)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{args.simulation_name}_{ts}.csv"
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False)

    print(f"Simulation saved: {out_path}")


if __name__ == "__main__":
    main()
