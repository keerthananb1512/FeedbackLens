"""Run each model in a fresh process on the same physical GPU."""

from pathlib import Path
import argparse
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen_json", "d1", "d3", "decision"],
        default=["qwen_json", "d1", "d3", "decision"],
    )
    parser.add_argument("--output", default=str(ROOT / "results"))
    args = parser.parse_args()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"}
    (ROOT / "logs").mkdir(exist_ok=True)
    for model in args.models:
        python = ROOT / (".venv-d1/bin/python" if model == "d1" else ".venv/bin/python")
        command = [
            str(python),
            "-m",
            "feedbacklens",
            "benchmark",
            "--model",
            model,
            "--output",
            args.output,
        ]
        print("Starting", model, "on physical GPU", args.gpu, flush=True)
        with (ROOT / f"logs/benchmark-{model}.log").open("w") as log:
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print("Finished", model, flush=True)
    if set(args.models) == {"qwen_json", "d1", "d3", "decision"}:
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "feedbacklens",
                "summarize",
                "--output",
                args.output,
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
