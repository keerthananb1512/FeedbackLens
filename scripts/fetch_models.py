"""Resolve immutable revisions, then download the exact requested model files."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi, snapshot_download
from feedbacklens.common import ROOT, write_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "configs/models.json"
    models = {"qwen_json": "unsloth/Qwen3.5-0.8B", "d1": "LiquidAI/d1-2B", "d3": "vllm-sr/d3-nano"}
    refs = (
        read_json(manifest)
        if manifest.exists() and not args.refresh
        else {
            key: {"repo_id": repo, "revision": HfApi().model_info(repo).sha}
            for key, repo in models.items()
        }
    )
    write_json(manifest, refs)
    for key, ref in refs.items():
        print("Downloading", key, ref, flush=True)
        folder = snapshot_download(
            **ref,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "*.jinja",
                "*.txt",
                "*.model",
                "LICENSE*",
                "README.md",
            ],
        )
        print(key, folder, flush=True)


if __name__ == "__main__":
    main()
