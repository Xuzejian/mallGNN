import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from mallgnn.data import check_temporal_splits, load_snapshot
from mallgnn.features import FeatureEncoder, feature_frame
from mallgnn.graph import RelationEdges, build_graph
from mallgnn.metrics import choose_threshold
from mallgnn.model import RelationalSAGELayer, ShopRiskModel
from mallgnn.pipeline import predict, train
from mallgnn.schema import RELATION_TYPES
from mallgnn.synthetic import generate_demo


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generate_demo(self.root / "data", num_shops=80, seed=42)

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self, split="train"):
        return load_snapshot(self.root / "data" / split)

    def test_feature_encoder_unknown_category_zero_orders_missing_statistics(self):
        data = self.snapshot()
        train_frame = feature_frame(data)
        encoder = FeatureEncoder().fit(train_frame)
        saved_state = copy.deepcopy(encoder.state)
        data.shops.loc[0, "category"] = "brand_new_category"
        data.stats.loc[:, "orders_7d"] = 0
        data.stats.loc[:, "orders_30d"] = 0
        data.stats = data.stats.iloc[1:]
        values = encoder.transform(feature_frame(data))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.shape, (80, len(encoder.feature_names)))
        self.assertEqual(values[0, encoder.feature_names.index("category:UNKNOWN")], 1)
        self.assertEqual(encoder.state, saved_state)
        self.assertNotIn("shop_id", encoder.feature_names)
        self.assertNotIn("label", encoder.feature_names)

    def test_future_feature_and_overlapping_snapshots_rejected(self):
        path = self.root / "data/train/shops.csv"
        shops = pd.read_csv(path)
        shops.loc[0, "feature_time"] = "2099-01-01"
        shops.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "after as_of"):
            self.snapshot()
        train_data, valid_data, test_data = self.snapshot("valid"), self.snapshot("valid"), self.snapshot("test")
        with self.assertRaisesRegex(ValueError, "temporal leakage"):
            check_temporal_splits(train_data, valid_data, test_data)

    def test_label_maturity_checked_before_next_snapshot(self):
        earlier, later, test = self.snapshot(), self.snapshot("valid"), self.snapshot("test")
        earlier.label_available_at[:] = pd.Timestamp("2026-03-15", tz="UTC")
        with self.assertRaisesRegex(ValueError, "temporal leakage"):
            check_temporal_splits(earlier, later, test)

    def test_graph_future_filter_duplicate_coalescing_and_degree_cap(self):
        data = self.snapshot()
        a, b, c = data.ids[:3]
        data.relations = pd.DataFrame([
            [a, b, "same_owner", 1.0, "2026-01-01"],
            [b, a, "same_owner", 1.0, "2026-01-01"],
            [a, b, "same_owner", 1.0, "2026-01-01"],
            [a, a, "same_owner", 1.0, "2026-01-01"],
            [c, b, "same_owner", 0.5, "2026-01-01"],
            [a, c, "same_device", 1.0, "2099-01-01"],
        ], columns=data.relations.columns)
        graph, info = build_graph(data, RELATION_TYPES, max_neighbors=1)
        edges = graph["same_owner"]
        self.assertEqual(info["future_rows_filtered"], 1)
        self.assertEqual(len(edges.src), 3)
        self.assertLessEqual(int(torch.bincount(edges.dst).max()), 1)
        self.assertEqual(len(graph["same_device"].src), 0)
        self.assertEqual(int(edges.src[edges.dst == 1][0]), 0)
        data.relations = data.relations.sample(frac=1, random_state=3)
        reordered, _ = build_graph(data, RELATION_TYPES, max_neighbors=1)
        self.assertTrue(torch.equal(edges.src, reordered["same_owner"].src))

    def test_unknown_endpoint_bad_weight_and_unknown_type_fail(self):
        for column, value, message in [("src_shop_id", "missing_shop", "unknown shop_id"),
                                        ("weight", -1.0, "weights"),
                                        ("relation_type", "unsupported", "unknown relation")]:
            data = self.snapshot()
            data.relations.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, message):
                build_graph(data, RELATION_TYPES)

    def test_weighted_mean_and_isolated_node(self):
        layer = RelationalSAGELayer(1, ["owner"])
        with torch.no_grad():
            layer.self_linear.weight.zero_()
            layer.self_linear.bias.zero_()
            layer.neighbor_linear["owner"].weight.fill_(1)
        h = torch.tensor([[2.0], [8.0], [0.0]])
        graph = {"owner": RelationEdges(torch.tensor([0, 1]), torch.tensor([2, 2]), torch.tensor([1.0, 3.0]))}
        result = layer(h, graph)
        self.assertAlmostEqual(result[2].item(), 6.5)
        self.assertEqual(result[0].item(), 0)

    def test_empty_stats_and_edgeless_graph_are_supported(self):
        data = self.snapshot()
        data.stats = data.stats.iloc[:0]
        data.relations = data.relations.iloc[:0]
        frame = feature_frame(data)
        x = torch.from_numpy(FeatureEncoder().fit(frame).transform(frame))
        graph, info = build_graph(data, RELATION_TYPES)
        self.assertTrue(torch.isfinite(x).all())
        self.assertEqual(sum(info["directed_edges"].values()), 0)
        model = ShopRiskModel(x.shape[1], hidden_dim=8, relation_types=RELATION_TYPES)
        self.assertTrue(torch.isfinite(model(x, graph)).all())

    def test_graph_model_node_permutation_equivariance_and_neighbor_gradients(self):
        torch.manual_seed(10)
        data = self.snapshot()
        encoder = FeatureEncoder().fit(feature_frame(data))
        x = torch.from_numpy(encoder.transform(feature_frame(data)))
        graph, _ = build_graph(data, RELATION_TYPES)
        model = ShopRiskModel(x.shape[1], hidden_dim=8, dropout=0, relation_types=RELATION_TYPES)
        model.eval()
        expected = model(x, graph)
        permutation = torch.randperm(len(x))
        inverse = torch.argsort(permutation)
        remapped = {kind: RelationEdges(inverse[edges.src], inverse[edges.dst], edges.weight) for kind, edges in graph.items()}
        actual = model(x[permutation], remapped)[inverse]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        expected.sum().backward()
        grad = model.layers[0].neighbor_linear["same_owner"].weight.grad
        self.assertGreater(grad.abs().sum().item(), 0)
        empty = {kind: RelationEdges(edges.src[:0], edges.dst[:0], edges.weight[:0]) for kind, edges in graph.items()}
        self.assertFalse(torch.allclose(model(x, empty), expected))

    def test_threshold_matches_brute_force_validation_f1(self):
        from sklearn.metrics import f1_score
        y = np.array([1, 0, 1, 1, 0, 0])
        p = np.array([0.1, 0.2, 0.4, 0.4, 0.8, 0.9])
        best = max((f1_score(y, p >= t), t) for t in np.unique(p))[1]
        self.assertEqual(choose_threshold(y, p), best)

    def test_train_reload_and_label_free_inference(self):
        output = self.root / "run"
        # An unlabeled node stays in the graph but is excluded from supervised loss.
        label_path = self.root / "data/train/labels.csv"
        labels = pd.read_csv(label_path)
        labels.iloc[1:].to_csv(label_path, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train(self.root / "data/train", self.root / "data/valid", self.root / "data/test", output,
                            epochs=4, hidden_dim=8, patience=3, threads=1)
        self.assertEqual(metrics["train"]["shops"], 79)
        checkpoint = torch.load(output / "model.pt", weights_only=True)
        encoder = FeatureEncoder(checkpoint["encoder"])
        expected_encoder = FeatureEncoder().fit(feature_frame(self.snapshot()))
        self.assertEqual(encoder.state, expected_encoder.state)
        # A malformed labels file must have no effect on prediction.
        (self.root / "data/test/labels.csv").write_text("invalid labels must be ignored\n")
        path = self.root / "scores.csv"
        predict(self.root / "data/test", output / "model.pt", path, threads=1)
        scores = pd.read_csv(path).set_index("shop_id").sort_index()
        original = pd.read_csv(output / "test_predictions.csv").set_index("shop_id").sort_index()
        np.testing.assert_allclose(scores["risk_score"], original["risk_score"], rtol=1e-6)
        self.assertEqual(len(scores), 80)
        self.assertTrue(scores["risk_score"].between(0, 1).all())
        np.testing.assert_array_equal(scores["prediction"], (scores["risk_score"] >= scores["threshold"]).astype(int))
        with self.assertRaisesRegex(ValueError, "predates model"):
            predict(self.root / "data/train", output / "model.pt", self.root / "past.csv", threads=1)
        # No trainable per-shop embeddings: unseen IDs can be scored without retraining.
        shop_path = self.root / "data/test/shops.csv"
        shops = pd.read_csv(shop_path)
        newcomer = shops.iloc[[0]].copy()
        newcomer["shop_id"] = "brand_new_isolated_shop"
        pd.concat([shops, newcomer]).to_csv(shop_path, index=False)
        result = predict(self.root / "data/test", output / "model.pt", self.root / "new_scores.csv", threads=1)
        self.assertEqual(result["shops"], 81)


if __name__ == "__main__":
    unittest.main()
