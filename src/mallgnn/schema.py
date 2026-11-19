"""Explicit feature allowlist: IDs, timestamps and future labels never enter X."""

BASE_NUMERIC = ["shop_age_days", "deposit_amount", "merchant_level", "is_enterprise"]
CATEGORICAL = ["category", "province"]
STATS = [
    "orders_7d", "orders_30d", "gmv_30d",
    "late_dispatch_orders_7d", "late_dispatch_orders_30d",
    "cancel_orders_30d", "refund_orders_30d", "complaint_orders_30d",
    "historical_violation_orders_30d", "avg_dispatch_hours_30d",
]
DERIVED = [
    "late_dispatch_rate_7d", "late_dispatch_rate_30d", "cancel_rate_30d",
    "refund_rate_30d", "complaint_rate_30d", "historical_violation_rate_30d",
    "orders_trend_7d_vs_30d", "avg_order_value_30d", "has_stats",
]
NUMERIC = BASE_NUMERIC + STATS + DERIVED
LOG_COLUMNS = [
    "shop_age_days", "deposit_amount", "orders_7d", "orders_30d", "gmv_30d",
    "late_dispatch_orders_7d", "late_dispatch_orders_30d", "cancel_orders_30d",
    "refund_orders_30d", "complaint_orders_30d", "historical_violation_orders_30d",
    "avg_dispatch_hours_30d", "orders_trend_7d_vs_30d", "avg_order_value_30d",
]
RELATION_TYPES = ["same_owner", "same_device", "same_warehouse"]
