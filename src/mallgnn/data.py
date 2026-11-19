"""CSV contract and point-in-time checks for a single graph snapshot."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .schema import BASE_NUMERIC, CATEGORICAL, STATS


def timestamp(value):
    return pd.to_datetime(value, utc=True, errors="raise", format="mixed")


def require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def check_unique(frame, column, name):
    if frame[column].isna().any() or frame[column].str.strip().eq("").any():
        raise ValueError(f"{name}.{column}: missing/empty ID")
    if frame[column].duplicated().any():
        raise ValueError(f"{name}.{column}: duplicate ID")


def check_times(frame, column, as_of, name):
    parsed = timestamp(frame[column])
    if parsed.isna().any() or (parsed > as_of).any():
        raise ValueError(f"{name}.{column}: missing timestamp or after as_of={as_of}")
    return parsed


@dataclass
class Snapshot:
    directory: Path
    as_of: pd.Timestamp
    horizon: int
    shops: pd.DataFrame
    stats: pd.DataFrame
    relations: pd.DataFrame
    labels: np.ndarray
    label_available_at: pd.Series

    @property
    def ids(self):
        return self.shops["shop_id"].tolist()

    @property
    def labeled(self):
        return self.labels >= 0


def load_snapshot(directory, require_labels=True):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise ValueError("metadata.schema_version must be 1")
    as_of = timestamp(meta["as_of"])
    horizon = meta["label_horizon_days"]
    if pd.isna(as_of) or type(horizon) is not int or horizon <= 0:
        raise ValueError("metadata requires valid as_of and positive integer label_horizon_days")
    shops = pd.read_csv(directory / "shops.csv", dtype={"shop_id": str})
    stats = pd.read_csv(directory / "stats.csv", dtype={"shop_id": str})
    relations = pd.read_csv(directory / "relations.csv", dtype={
        "src_shop_id": str, "dst_shop_id": str, "relation_type": str,
    })
    require_columns(shops, ["shop_id", "feature_time"] + BASE_NUMERIC + CATEGORICAL, "shops")
    require_columns(stats, ["shop_id", "feature_time"] + STATS, "stats")
    require_columns(relations, ["src_shop_id", "dst_shop_id", "relation_type", "weight", "observed_at"], "relations")
    check_unique(shops, "shop_id", "shops")
    check_unique(stats, "shop_id", "stats")
    if shops.empty:
        raise ValueError("shops.csv must contain at least one shop")
    ids = set(shops["shop_id"])
    if not set(stats["shop_id"]).issubset(ids):
        raise ValueError("stats references unknown shop_id")
    check_times(shops, "feature_time", as_of, "shops")
    check_times(stats, "feature_time", as_of, "stats")
    for name, frame, columns in [("shops", shops, BASE_NUMERIC), ("stats", stats, STATS)]:
        for col in columns:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            non_missing = frame[col].dropna()
            if not np.isfinite(non_missing).all() or (non_missing < 0).any():
                raise ValueError(f"{name}.{col} must be nonnegative and finite (NaN allowed)")
    if not shops["is_enterprise"].dropna().isin([0, 1]).all():
        raise ValueError("is_enterprise must be 0/1")
    for col in ["late_dispatch_orders_7d", "late_dispatch_orders_30d", "cancel_orders_30d",
                "refund_orders_30d", "complaint_orders_30d", "historical_violation_orders_30d"]:
        denominator = "orders_7d" if col.endswith("7d") else "orders_30d"
        if (stats[col] > stats[denominator]).any():
            raise ValueError(f"{col} cannot exceed {denominator}")
    if (stats["orders_7d"] > stats["orders_30d"]).any():
        raise ValueError("orders_7d cannot exceed orders_30d")
    if (stats["late_dispatch_orders_7d"] > stats["late_dispatch_orders_30d"]).any():
        raise ValueError("late_dispatch_orders_7d cannot exceed late_dispatch_orders_30d")

    y = np.full(len(shops), -1, dtype=np.float32)
    available = pd.Series([], dtype="datetime64[ns, UTC]")
    # Prediction deliberately does not open labels.csv, even if one is present.
    if require_labels:
        labels = pd.read_csv(directory / "labels.csv", dtype={"shop_id": str})
        require_columns(labels, ["shop_id", "label", "label_start", "label_end", "label_available_at"], "labels")
        check_unique(labels, "shop_id", "labels")
        if not set(labels["shop_id"]).issubset(ids):
            raise ValueError("labels references unknown shop_id")
        if not labels["label"].isin([0, 1]).all():
            raise ValueError("label must be 0/1; omit the row for an unlabeled shop")
        start, end = timestamp(labels["label_start"]), timestamp(labels["label_end"])
        available = timestamp(labels["label_available_at"])
        expected_end = as_of + pd.Timedelta(days=horizon)
        if (start.isna().any() or end.isna().any() or available.isna().any()
                or (start != as_of).any() or (end != expected_end).any()
                or (available < end).any()):
            raise ValueError("labels must describe (as_of, as_of + horizon] and be available after label_end")
        mapped = labels.set_index("shop_id")["label"]
        y = shops["shop_id"].map(mapped).fillna(-1).to_numpy(dtype=np.float32)
        if not (y >= 0).any():
            raise ValueError("snapshot has no labeled shops")
    return Snapshot(directory, as_of, horizon, shops, stats, relations, y, available)


def check_temporal_splits(train, valid, test):
    if len({x.horizon for x in [train, valid, test]}) != 1:
        raise ValueError("all splits must use the same label horizon")
    for earlier, later in [(train, valid), (valid, test)]:
        if earlier.as_of >= later.as_of or earlier.label_available_at.max() > later.as_of:
            raise ValueError("temporal leakage: earlier labels are not available by next snapshot as_of")
