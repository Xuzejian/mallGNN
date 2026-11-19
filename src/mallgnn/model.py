"""Weighted, relation-specific mean GraphSAGE in plain PyTorch (no PyG)."""

import torch
from torch import nn
import torch.nn.functional as F


class RelationalSAGELayer(nn.Module):
    def __init__(self, hidden_dim, relation_types):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in relation_types})

    def forward(self, h, graph):
        message = torch.zeros_like(h)
        present = h.new_zeros((len(h), 1))
        for relation, linear in self.neighbor_linear.items():
            edges = graph[relation]
            if not edges.src.numel():
                continue
            total = torch.zeros_like(h)
            denominator = h.new_zeros((len(h), 1))
            weight = edges.weight[:, None]
            total.index_add_(0, edges.dst, h[edges.src] * weight)
            denominator.index_add_(0, edges.dst, weight)
            mean = total / denominator.clamp_min(1e-12)
            message = message + linear(mean)
            present = present + (denominator > 0).to(h.dtype)
        return F.relu(self.self_linear(h) + message / present.clamp_min(1))


class ShopRiskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, num_layers=2, dropout=0.15,
                 relation_types=None, model_type="graphsage"):
        super().__init__()
        if model_type not in {"graphsage", "mlp"}:
            raise ValueError("model_type must be graphsage or mlp")
        self.model_type, self.dropout = model_type, dropout
        self.encoder = nn.Linear(input_dim, hidden_dim)
        if model_type == "graphsage":
            self.layers = nn.ModuleList([RelationalSAGELayer(hidden_dim, relation_types) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, graph):
        own = F.relu(self.encoder(x))
        h = own
        for layer in self.layers:
            h = layer(h, graph) if self.model_type == "graphsage" else F.relu(layer(h))
            h = F.dropout(h, self.dropout, training=self.training)
        # Preserve own-shop information even with noisy/high-degree neighbors.
        return self.head(torch.cat([own, h], dim=-1)).squeeze(-1)
