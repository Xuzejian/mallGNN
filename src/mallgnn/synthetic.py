"""Reproducible, fictitious samples. Latent risks are never exported as features."""

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def generate_demo(output_dir, num_shops=1000, seed=42):
    if num_shops < 30:
        raise ValueError("num_shops must be >= 30 to create meaningful binary splits")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / split).exists() for split in ["train", "valid", "test"]):
        raise ValueError(f"{root} already contains snapshots; choose a new output directory")
    rng = np.random.default_rng(seed)
    n = num_shops
    ids = np.array([f"shop_{i:05d}" for i in range(n)])
    owner = rng.permutation(n) // 5
    device = rng.permutation(n) // 4
    warehouse = rng.integers(0, max(3, n // 25), n)
    owner_risk = rng.normal(0, 1, int(owner.max()) + 1)
    device_risk = rng.normal(0, 1, int(device.max()) + 1)
    individual = rng.normal(0, 0.8, n)
    age = rng.integers(15, 1800, n)
    categories = rng.choice(["apparel", "electronics", "home", "beauty"], n)
    provinces = rng.choice(["Shanghai", "Zhejiang", "Guangdong", "Jiangsu"], n)
    enterprise = rng.binomial(1, 0.65, n)
    level = rng.integers(1, 6, n)
    deposit = rng.choice([1000, 3000, 5000, 10000, 30000], n)
    summaries = {}
    for split, date in [("train", "2026-01-31"), ("valid", "2026-02-28"), ("test", "2026-03-31")]:
        directory = root / split
        directory.mkdir()
        as_of = pd.Timestamp(date, tz="UTC")
        # All behavior is generated before sampling future outcomes.
        behavior = 0.55 * owner_risk[owner] + individual + rng.normal(0, 0.6, n)
        orders30 = np.maximum(1, rng.lognormal(4.8, 0.8, n).astype(int))
        orders7 = rng.binomial(orders30, 7 / 30)
        late7 = rng.binomial(orders7, 0.008 + 0.19 * sigmoid(behavior - 0.7))
        late30 = late7 + rng.binomial(orders30 - orders7, 0.008 + 0.19 * sigmoid(behavior - 0.7))
        shops = pd.DataFrame({
            "shop_id": ids, "feature_time": date,
            "shop_age_days": age + (as_of - pd.Timestamp("2026-01-31", tz="UTC")).days,
            "category": categories, "province": provinces, "merchant_level": level,
            "deposit_amount": deposit, "is_enterprise": enterprise,
        })
        stats = pd.DataFrame({
            "shop_id": ids, "feature_time": date,
            "orders_7d": orders7, "orders_30d": orders30,
            "gmv_30d": (orders30 * rng.lognormal(4.4, 0.4, n)).round(2),
            "late_dispatch_orders_7d": late7, "late_dispatch_orders_30d": late30,
            "cancel_orders_30d": rng.binomial(orders30, 0.01 + 0.06 * sigmoid(behavior)),
            "refund_orders_30d": rng.binomial(orders30, 0.015 + 0.08 * sigmoid(behavior)),
            "complaint_orders_30d": rng.binomial(orders30, 0.002 + 0.025 * sigmoid(behavior)),
            "historical_violation_orders_30d": rng.binomial(orders30, 0.001 + 0.035 * sigmoid(behavior)),
            "avg_dispatch_hours_30d": (12 + 30 * sigmoid(behavior) + rng.uniform(0, 8, n)).round(2),
        })
        # A few cold-start shops have no statistics. One value is missing on purpose.
        stats = stats.loc[rng.random(n) > 0.03].copy()
        shops.loc[0, "deposit_amount"] = np.nan
        relation_rows = []
        for kind, groups, weight in [("same_owner", owner, 1.0), ("same_device", device, 0.8), ("same_warehouse", warehouse, 0.3)]:
            for group in np.unique(groups):
                members = ids[groups == group]
                # Limit noisy warehouse hubs at the source too.
                for src, dst in combinations(members[:30], 2):
                    relation_rows.append((src, dst, kind, weight, date))
        relations = pd.DataFrame(relation_rows, columns=["src_shop_id", "dst_shop_id", "relation_type", "weight", "observed_at"])
        future_probability = sigmoid(-2.2 + 1.0 * behavior + 1.35 * owner_risk[owner] + 0.5 * device_risk[device])
        labels = rng.binomial(1, future_probability)
        if len(np.unique(labels)) != 2:
            raise ValueError("this seed generated a single-class split; use another seed or more shops")
        end = as_of + pd.Timedelta(days=7)
        target = pd.DataFrame({
            "shop_id": ids, "label": labels, "label_start": date,
            "label_end": end.date().isoformat(),
            "label_available_at": (end + pd.Timedelta(days=2)).date().isoformat(),
        })
        for name, frame in [("shops", shops), ("stats", stats), ("relations", relations), ("labels", target)]:
            frame.to_csv(directory / f"{name}.csv", index=False)
        (directory / "metadata.json").write_text(json.dumps({
            "schema_version": 1, "as_of": date, "label_horizon_days": 7,
            "synthetic": True, "seed": seed,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries[split] = {"shops": n, "undirected_relations": len(relations), "positives": int(labels.sum()), "positive_rate": float(labels.mean())}
    return summaries
