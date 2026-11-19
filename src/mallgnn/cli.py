import argparse
from pathlib import Path
import json

from .pipeline import predict, train, write_json
from .synthetic import generate_demo


def main():
    parser = argparse.ArgumentParser(description="履约违规店铺预测：基础特征 + 多关系 GraphSAGE")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("generate", help="生成完全虚构的三个时间快照")
    sample.add_argument("--output-dir", default="data/demo")
    sample.add_argument("--num-shops", type=int, default=1000)
    sample.add_argument("--seed", type=int, default=42)
    fit = commands.add_parser("train", help="按时间快照训练/验证/测试")
    fit.add_argument("--train-dir", required=True)
    fit.add_argument("--valid-dir", required=True)
    fit.add_argument("--test-dir", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--model-type", choices=["graphsage", "mlp"], default="graphsage")
    for name, default in [("epochs", 120), ("hidden-dim", 48), ("num-layers", 2), ("patience", 20), ("max-neighbors", 30), ("seed", 42), ("threads", 2)]:
        fit.add_argument(f"--{name}", type=int, default=default)
    for name, default in [("dropout", 0.15), ("lr", 0.005), ("weight-decay", 0.0001)]:
        fit.add_argument(f"--{name}", type=float, default=default)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--relation-types", nargs="+", default=None)
    infer = commands.add_parser("predict", help="无需标签的批量打分")
    infer.add_argument("--data-dir", required=True)
    infer.add_argument("--checkpoint-path", required=True)
    infer.add_argument("--output-path", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--threads", type=int, default=2)
    demo = commands.add_parser("demo", help="一键生成样本、对比 MLP/GraphSAGE、加载模型预测")
    demo.add_argument("--work-dir", default="artifacts/demo")
    demo.add_argument("--num-shops", type=int, default=1000)
    demo.add_argument("--epochs", type=int, default=120)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--threads", type=int, default=2)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "generate":
            result = generate_demo(**args)
        elif command == "train":
            result = train(**args)
        elif command == "predict":
            result = predict(**args)
        else:
            root = Path(args["work_dir"])
            if root.exists() and any(root.iterdir()):
                raise ValueError(f"{root} is not empty; choose a new --work-dir")
            summary = generate_demo(root / "data", args["num_shops"], args["seed"])
            result = {"samples": summary, "test_metrics": {}}
            for model_type in ["mlp", "graphsage"]:
                metrics = train(root / "data/train", root / "data/valid", root / "data/test", root / model_type,
                                model_type=model_type, epochs=args["epochs"], seed=args["seed"], threads=args["threads"])
                result["test_metrics"][model_type] = metrics["test"]
            result["prediction"] = predict(root / "data/test", root / "graphsage/model.pt", root / "risk_scores.csv", threads=args["threads"])
            write_json(root / "comparison.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
