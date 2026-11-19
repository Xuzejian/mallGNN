"""End-to-end training and label-free inference on new graph snapshots."""

from pathlib import Path
import json
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import __version__
from .data import check_temporal_splits, load_snapshot
from .features import FeatureEncoder, feature_frame
from .graph import build_graph
from .metrics import choose_threshold, evaluate
from .model import ShopRiskModel
from .schema import RELATION_TYPES


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_seed(seed, threads):
    if threads < 1:
        raise ValueError("threads must be >= 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def prepare(snapshot, encoder, relation_types, max_neighbors, device):
    x = torch.from_numpy(encoder.transform(feature_frame(snapshot))).to(device)
    graph, info = build_graph(snapshot, relation_types, max_neighbors)
    graph = {relation: edges.to(device) for relation, edges in graph.items()}
    return x, graph, torch.tensor(snapshot.labels, device=device), torch.tensor(snapshot.labeled, device=device), info


@torch.no_grad()
def probabilities(model, batch):
    model.eval()
    # Keep exact float32 values in float64 CSV columns, including threshold ties.
    return model(batch[0], batch[1]).sigmoid().cpu().numpy().astype(np.float64)


def train(train_dir, valid_dir, test_dir, output_dir, model_type="graphsage",
          epochs=120, hidden_dim=48, num_layers=2, dropout=0.15, lr=0.005,
          weight_decay=0.0001, patience=20, max_neighbors=30, seed=42,
          threads=2, device="cpu", relation_types=None):
    relation_types = RELATION_TYPES if relation_types is None else relation_types
    if not relation_types or len(set(relation_types)) != len(relation_types) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", r) for r in relation_types):
        raise ValueError("relation types must be unique lowercase identifiers")
    if epochs < 1 or hidden_dim < 1 or num_layers < 1 or patience < 1 or lr <= 0 or weight_decay < 0 or not 0 <= dropout < 1:
        raise ValueError("invalid training hyperparameters")
    set_seed(seed, threads)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    train_data, valid_data, test_data = [load_snapshot(path) for path in [train_dir, valid_dir, test_dir]]
    check_temporal_splits(train_data, valid_data, test_data)
    for name, snapshot in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        if len(np.unique(snapshot.labels[snapshot.labeled])) != 2:
            raise ValueError(f"{name} must contain both positive and negative labels")
    # Fit preprocessing on the historical training snapshot only, including its unlabeled nodes.
    encoder = FeatureEncoder().fit(feature_frame(train_data))
    batches = [prepare(snapshot, encoder, relation_types, max_neighbors, device) for snapshot in [train_data, valid_data, test_data]]
    train_batch, valid_batch, test_batch = batches
    config = dict(input_dim=train_batch[0].shape[1], hidden_dim=hidden_dim, num_layers=num_layers,
                  dropout=dropout, relation_types=relation_types, model_type=model_type)
    model = ShopRiskModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = train_batch[2][train_batch[3]]
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    history, best_score, best_state, best_epoch, stale = [], -float("inf"), None, 0, 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_batch[0], train_batch[1])
        loss = criterion(logits[train_batch[3]], y)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        val_p = probabilities(model, valid_batch)[valid_data.labeled]
        val_metrics = evaluate(valid_data.labels[valid_data.labeled], val_p, 0.5)
        score = val_metrics["pr_auc"]
        history.append({"epoch": epoch, "loss": float(loss.detach()), "valid_pr_auc": score})
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{model_type}: epoch={epoch:03d} loss={loss.item():.4f} valid_pr_auc={score:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    val_p = probabilities(model, valid_batch)[valid_data.labeled]
    threshold = choose_threshold(valid_data.labels[valid_data.labeled], val_p)
    # Test labels are used only after model selection and threshold selection are finished.
    predictions = {name: probabilities(model, batch) for name, batch in zip(["train", "valid", "test"], batches)}
    metrics = {name: evaluate(snapshot.labels[snapshot.labeled], predictions[name][snapshot.labeled], threshold)
               for name, snapshot in zip(["train", "valid", "test"], [train_data, valid_data, test_data])}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1, "mallgnn_version": __version__, "model_config": config,
        "model_state": best_state, "encoder": encoder.state, "threshold": threshold,
        "max_neighbors": max_neighbors, "train_as_of": train_data.as_of.isoformat(),
        "validation_as_of": valid_data.as_of.isoformat(),
        "model_available_at": valid_data.label_available_at.max().isoformat(),
        "label_horizon_days": train_data.horizon,
    }
    torch.save(checkpoint, output / "model.pt")
    write_json(output / "metrics.json", metrics)
    write_json(output / "history.json", history)
    write_json(output / "feature_names.json", encoder.feature_names)
    write_json(output / "run.json", {
        "model_config": config, "seed": seed, "best_epoch": best_epoch,
        "epochs_run": len(history), "seconds": round(time.monotonic() - started, 3),
        "positive_weight": positive_weight, "threshold_selection": "max validation F1",
        "graph": {name: batch[4] for name, batch in zip(["train", "valid", "test"], batches)},
        "snapshots": {name: {"as_of": snapshot.as_of.isoformat(), "label_available_at": snapshot.label_available_at.max().isoformat()}
                      for name, snapshot in zip(["train", "valid", "test"], [train_data, valid_data, test_data])},
        "hyperparameters": dict(epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience,
                                max_neighbors=max_neighbors, threads=threads, device=device),
        "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
    })
    for name, snapshot in zip(["train", "valid", "test"], [train_data, valid_data, test_data]):
        pd.DataFrame({"shop_id": snapshot.ids, "label": snapshot.labels.astype(int),
                      "risk_score": predictions[name], "prediction": (predictions[name] >= threshold).astype(int)}).sort_values("risk_score", ascending=False).to_csv(output / f"{name}_predictions.csv", index=False)
    return metrics


def predict(data_dir, checkpoint_path, output_path, device="cpu", threads=2):
    set_seed(0, threads)
    # weights_only avoids executable pickle objects; load models produced by this project.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    snapshot = load_snapshot(data_dir, require_labels=False)
    if snapshot.horizon != checkpoint["label_horizon_days"]:
        raise ValueError("prediction label horizon differs from trained model")
    if snapshot.as_of < pd.Timestamp(checkpoint["model_available_at"]):
        raise ValueError("prediction as_of predates model/threshold availability")
    encoder = FeatureEncoder(checkpoint["encoder"])
    config = checkpoint["model_config"]
    batch = prepare(snapshot, encoder, config["relation_types"], checkpoint["max_neighbors"], device)
    model = ShopRiskModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probability = probabilities(model, batch)
    result = pd.DataFrame({
        "shop_id": snapshot.ids, "as_of": snapshot.as_of.isoformat(),
        "risk_score": probability, "prediction": (probability >= checkpoint["threshold"]).astype(int),
        "threshold": checkpoint["threshold"],
    }).sort_values("risk_score", ascending=False, kind="stable")
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"{output} already exists; choose another output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return {"shops": len(result), "flagged": int(result["prediction"].sum()), "output": str(output), "graph": batch[4]}
