"""Sparse typed graphs; O(N + E), with no N x N adjacency matrix."""

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch

from .data import timestamp


@dataclass
class RelationEdges:
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor

    def to(self, device):
        return RelationEdges(self.src.to(device), self.dst.to(device), self.weight.to(device))


def build_graph(snapshot, relation_types, max_neighbors=30):
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be >= 1")
    frame = snapshot.relations.copy()
    observed = timestamp(frame["observed_at"])
    if observed.isna().any():
        raise ValueError("relations.observed_at cannot be missing")
    future = observed > snapshot.as_of
    future_count = int(future.sum())
    frame = frame.loc[~future].copy()
    if frame[["src_shop_id", "dst_shop_id", "relation_type"]].isna().any().any():
        raise ValueError("relation endpoints and type cannot be missing")
    unknown_types = set(frame["relation_type"]) - set(relation_types)
    if unknown_types:
        raise ValueError(f"unknown relation types: {sorted(unknown_types)}")
    index = {shop: i for i, shop in enumerate(snapshot.ids)}
    if not (set(frame["src_shop_id"]) | set(frame["dst_shop_id"])).issubset(index):
        raise ValueError("relations references unknown shop_id")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if not np.isfinite(frame["weight"]).all() or (frame["weight"] <= 0).any():
        raise ValueError("relation weights must be finite and > 0")
    graph, counts = {}, {}
    for relation in relation_types:
        # Max coalescing avoids counting repeated rows or reciprocal edges twice.
        pairs = {}
        for src, dst, weight in frame.loc[frame["relation_type"].eq(relation), ["src_shop_id", "dst_shop_id", "weight"]].itertuples(index=False, name=None):
            if src == dst:
                continue
            for a, b in [(src, dst), (dst, src)]:
                pairs[(a, b)] = max(pairs.get((a, b), 0), weight)
        incoming = {}
        for (src, dst), weight in pairs.items():
            # Hash tie-breaking is stable across input order and shop row order.
            tie = hashlib.sha256(f"{relation}|{src}|{dst}".encode()).hexdigest()
            incoming.setdefault(dst, []).append((src, weight, tie))
        kept = []
        for dst in sorted(incoming):
            candidates = sorted(incoming[dst], key=lambda row: (-row[1], row[2]))
            kept.extend((index[src], index[dst], weight) for src, weight, _ in candidates[:max_neighbors])
        graph[relation] = RelationEdges(
            torch.tensor([row[0] for row in kept], dtype=torch.long),
            torch.tensor([row[1] for row in kept], dtype=torch.long),
            torch.tensor([row[2] for row in kept], dtype=torch.float32),
        )
        counts[relation] = len(kept)
    return graph, {"nodes": len(index), "directed_edges": counts, "future_rows_filtered": future_count, "max_neighbors_per_relation": max_neighbors}
