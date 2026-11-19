"""Train-fitted numerical imputation/scaling and categorical vocabularies."""

import numpy as np
import pandas as pd

from .schema import CATEGORICAL, LOG_COLUMNS, NUMERIC, STATS


def feature_frame(snapshot):
    merged = snapshot.shops.merge(snapshot.stats[["shop_id"] + STATS], on="shop_id", how="left", validate="one_to_one", indicator=True)
    merged["has_stats"] = merged["_merge"].eq("both").astype(float)

    def rate(count, denominator):
        # With zero orders a rate is unknown, not zero; training medians handle it.
        return merged[count] / merged[denominator].replace(0, np.nan)

    for name, count, denominator in [
        ("late_dispatch_rate_7d", "late_dispatch_orders_7d", "orders_7d"),
        ("late_dispatch_rate_30d", "late_dispatch_orders_30d", "orders_30d"),
        ("cancel_rate_30d", "cancel_orders_30d", "orders_30d"),
        ("refund_rate_30d", "refund_orders_30d", "orders_30d"),
        ("complaint_rate_30d", "complaint_orders_30d", "orders_30d"),
        ("historical_violation_rate_30d", "historical_violation_orders_30d", "orders_30d"),
    ]:
        merged[name] = rate(count, denominator)
    merged["orders_trend_7d_vs_30d"] = rate("orders_7d", "orders_30d") * 30 / 7
    merged["avg_order_value_30d"] = rate("gmv_30d", "orders_30d")
    return merged[NUMERIC + CATEGORICAL].copy()


class FeatureEncoder:
    def __init__(self, state=None):
        self.state = state

    @staticmethod
    def numeric(frame):
        x = frame[NUMERIC].astype(float).copy()
        x[LOG_COLUMNS] = np.log1p(x[LOG_COLUMNS])
        return x.replace([np.inf, -np.inf], np.nan)

    def fit(self, frame):
        x = self.numeric(frame)
        median = x.median().fillna(0)
        x = x.fillna(median)
        mean, std = x.mean(), x.std(ddof=0).replace(0, 1).fillna(1)
        self.state = {
            "numeric_columns": NUMERIC,
            "median": median.tolist(), "mean": mean.tolist(), "std": std.tolist(),
            "vocab": {col: sorted(frame[col].dropna().astype(str).unique().tolist()) for col in CATEGORICAL},
        }
        return self

    def transform(self, frame):
        if self.state is None or self.state["numeric_columns"] != NUMERIC:
            raise ValueError("missing or incompatible feature encoder")
        raw = self.numeric(frame).to_numpy()
        missing = np.isnan(raw).astype(np.float32)
        filled = np.where(np.isnan(raw), np.asarray(self.state["median"]), raw)
        values = np.clip((filled - self.state["mean"]) / self.state["std"], -10, 10)
        blocks = [values, missing]
        for col in CATEGORICAL:
            vocabulary = self.state["vocab"][col]
            mapping = {value: i + 1 for i, value in enumerate(vocabulary)}
            indices = frame[col].map(lambda value: mapping.get(str(value), 0)).to_numpy()
            blocks.append(np.eye(len(vocabulary) + 1, dtype=np.float32)[indices])
        return np.concatenate(blocks, axis=1).astype(np.float32)

    @property
    def feature_names(self):
        names = NUMERIC + [f"missing:{col}" for col in NUMERIC]
        for col in CATEGORICAL:
            names += [f"{col}:UNKNOWN"] + [f"{col}:{value}" for value in self.state["vocab"][col]]
        return names
