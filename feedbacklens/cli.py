from __future__ import annotations

import argparse
import json
from .common import ROOT


def main():
    parser = argparse.ArgumentParser(prog="feedbacklens")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", default=str(ROOT / "data"))
    p.add_argument("--revision")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p = sub.add_parser("train")
    p.add_argument("--config", default=str(ROOT / "configs/train.json"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--output", default=str(ROOT / "artifacts/decision"))
    p.add_argument("--resume")
    p = sub.add_parser("benchmark")
    p.add_argument("--model", choices=("decision", "qwen_json", "d1", "d3"), required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--output", default=str(ROOT / "results"))
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--limit", type=int, help="Smoke test only; final reports require the full official test"
    )
    p = sub.add_parser("predict")
    p.add_argument("--model", choices=("decision", "qwen_json", "d1", "d3"), default="decision")
    p.add_argument("--text", required=True)
    p.add_argument("--checkpoint")
    p = sub.add_parser("verify-checkpoint")
    p.add_argument("--checkpoint")
    p.add_argument("--output", default=str(ROOT / "artifacts/decision"))
    p.add_argument("--report-path")
    p = sub.add_parser("summarize")
    p.add_argument("--output", default=str(ROOT / "results"))
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .prepare import prepare

        result = prepare(**args)
    elif command == "train":
        from .train import train

        args["config_path"] = args.pop("config")
        result = train(**args)
    elif command == "benchmark":
        from .evaluate import benchmark

        args["key"] = args.pop("model")
        result = benchmark(**args)
    elif command == "predict":
        from .inference import Backend

        text = args.pop("text")
        result = Backend(args.pop("model"), **args).predict(text).to_dict()
    elif command == "verify-checkpoint":
        from .evaluate import verify_checkpoint

        result = verify_checkpoint(**args)
    else:
        from .evaluate import summarize

        result = summarize(**args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
