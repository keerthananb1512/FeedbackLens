from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .common import (
    ROOT,
    digest,
    environment,
    file_digest,
    gpu_processes,
    read_json,
    read_jsonl,
    write_json,
)
from .schema import load_schema


def classification_metrics(gold, predicted, labels):
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Expected equally sized, nonempty gold and predictions")
    invalid = "__invalid__"
    predicted = [p if p in labels else invalid for p in predicted]
    p, r, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1.mean()),
        "invalid_predictions": predicted.count(invalid),
        "n": len(gold),
        "per_class": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_labels": labels + [invalid],
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels + [invalid]).tolist(),
    }


def benchmark(
    key,
    data_dir=ROOT / "data",
    output=ROOT / "results",
    checkpoint=None,
    warmups=10,
    batch_size=1,
    limit=None,
):
    if batch_size != 1:
        raise ValueError("This synchronous latency protocol uses batch_size=1 for every model")
    if warmups < 1:
        raise ValueError("At least one warmup is required")
    from .inference import Backend

    backend = Backend(key, checkpoint=checkpoint)
    torch = backend.torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA")
    schema = load_schema()
    labels = [r["label"] for r in schema]
    data_dir, output = Path(data_dir), Path(output)
    full_test = read_jsonl(data_dir / "test.jsonl")
    audit = read_json(data_dir / "audit.json")
    if (
        file_digest(data_dir / "test.jsonl") != audit["files"]["test"]
        or digest(full_test) != audit["official_test_sha256"]
    ):
        raise ValueError("Test data no longer matches the preserved official split")
    test = full_test[:limit] if limit is not None else full_test
    if not test:
        raise ValueError("Empty evaluation subset")
    output.mkdir(parents=True, exist_ok=True)
    start_environment = environment()
    # Identical validation messages warm all models; warmups are not scored or timed.
    warm_rows = read_jsonl(data_dir / "validation.jsonl")
    for i in range(warmups):
        backend.predict(warm_rows[i % len(warm_rows)]["text"])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    process_samples = []
    partial = output / f"{key}.predictions.partial.jsonl"
    started = time.perf_counter()
    with partial.open("w") as f:
        for i, row in enumerate(test):
            if i % 100 == 0:
                process_samples.append({"row": i, "processes": gpu_processes()})
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            try:
                prediction = backend.predict(row["text"]).to_dict()
            except (ValueError, KeyError, TypeError) as exc:
                # Malformed model outputs are incorrect, never silently dropped.
                prediction = {
                    "label": None,
                    "probabilities": None,
                    "raw": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            result = {
                **row,
                "gold_label": labels[row["label"]],
                **prediction,
                "latency_ms": elapsed,
            }
            predictions.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"{key}: {i + 1}/{len(test)}", flush=True)
    latency = np.array([r["latency_ms"] for r in predictions])
    report = {
        **backend.metadata,
        "metrics": classification_metrics(
            [r["gold_label"] for r in predictions], [r["label"] for r in predictions], labels
        ),
        "latency_ms": {
            "median": float(np.median(latency)),
            "p95": float(np.percentile(latency, 95)),
            "mean": float(latency.mean()),
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "PyTorch process allocator; includes resident model, excludes other processes and CUDA driver allocations",
        },
        "protocol": {
            "warmups": warmups,
            "batch_size": batch_size,
            "includes": [
                "tokenization",
                "model forward/generation",
                "output decoding/parsing",
                "probability conversion",
                "CUDA synchronization",
            ],
            "excludes": [
                "model loading",
                "warmup",
                "writing predictions",
                "scoring metrics",
                "nvidia-smi sampling",
            ],
            "timer": "perf_counter_ns with CUDA synchronize before start and before stop",
            "test_rows": len(test),
            "full_official_test": limit is None or len(test) == len(full_test),
            "test_sha256": file_digest(data_dir / "test.jsonl"),
            "evaluated_row_ids_sha256": digest([r["id"] for r in predictions]),
            "validation_warmup_sha256": digest(
                [warm_rows[i % len(warm_rows)]["id"] for i in range(warmups)]
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": time.perf_counter() - started,
        },
        "environment": start_environment,
        "environment_after": environment(),
        "gpu_process_samples": process_samples,
        "limitations": [
            "External workloads were not stopped; consult GPU process samples for contention.",
            "Zero-shot denotes no local BANKING77 tuning; upstream training-data contamination is unknown.",
            "Probabilities from different decision architectures are not interchangeable confidence guarantees.",
        ],
    }
    final = output / f"{key}.predictions.jsonl"
    partial.replace(final)
    report["predictions_sha256"] = file_digest(final)
    write_json(output / f"{key}.json", report)
    return report


def compare_tensor_states(expected, actual):
    """Check exact keys, dtypes, and values rather than inference rounding."""
    import torch

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    different = [
        name
        for name in sorted(set(expected) & set(actual))
        if expected[name].dtype != actual[name].dtype
        or not torch.equal(expected[name].cpu(), actual[name].detach().cpu())
    ]
    return {
        "tensors": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "different": different,
        "bit_identical": not (missing or unexpected or different),
    }


def verify_checkpoint(checkpoint=None, output=ROOT / "artifacts/decision", report_path=None):
    from .inference import Backend

    output = Path(output)
    checkpoint = Path(checkpoint or output / "best").resolve()
    backend = Backend("decision", checkpoint=checkpoint)
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    weights = {
        "head": compare_tensor_states(
            load_file(str(checkpoint / "joint_head.safetensors")),
            backend.model.head.state_dict(),
        ),
        "adapter": compare_tensor_states(
            load_file(str(checkpoint / "adapter_model.safetensors")),
            get_peft_model_state_dict(backend.model.encoder),
        ),
    }
    saved_config = read_json(checkpoint / "unsloth_decision_config.json")
    adapter_config = read_json(checkpoint / "adapter_config.json")
    configuration_match = (
        backend.model.decision_config == saved_config
        and saved_config["base_model"] == backend.ref["repo_id"]
        and saved_config["base_revision"] == backend.ref["revision"]
        and adapter_config["revision"] == backend.ref["revision"]
        and read_json(checkpoint / "intents.json") == backend.schema
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        backend.ref["repo_id"], revision=backend.ref["revision"], local_files_only=True
    )
    reference = read_json(output / "reload_reference.json")
    if not reference:
        raise ValueError("No pre-save validation probes")
    tokenizer_inputs = [p["text"] for p in reference] + [
        json.dumps(backend.questions, ensure_ascii=False)
    ]
    loaded_tokenizer = getattr(backend.tokenizer, "tokenizer", backend.tokenizer)
    tokenizer_match = all(
        loaded_tokenizer.encode(text) == base_tokenizer.encode(text) for text in tokenizer_inputs
    )
    comparisons = []
    for probe in reference:
        got = backend.predict(probe["text"])
        expected = probe["prediction"]
        before = np.array([expected["probabilities"][label] for label in backend.labels])
        after = np.array([got.probabilities[label] for label in backend.labels])
        comparisons.append(
            {
                "id": probe["id"],
                "expected_label": expected["answer"],
                "loaded_label": got.label,
                "max_probability_abs_error": float(abs(before - after).max()),
                "label_match": expected["answer"] == got.label,
                "probabilities_close": bool(np.allclose(before, after, atol=1e-4, rtol=1e-3)),
            }
        )
    ok = (
        all(w["bit_identical"] for w in weights.values())
        and configuration_match
        and tokenizer_match
        and all(r["label_match"] for r in comparisons)
    )
    files = {p.name: file_digest(p) for p in checkpoint.iterdir() if p.is_file()}
    result = {
        "passed": ok,
        "acceptance": "Exact restored head/adapter tensors and configuration, matching tokenizer probes, valid distributions and unchanged validation labels",
        "fresh_process_reload": True,
        "checkpoint": str(checkpoint),
        "weights": weights,
        "configuration_match": configuration_match,
        "tokenizer_probe_match": tokenizer_match,
        "comparisons": comparisons,
        "probability_reproducibility": {
            "all_close": all(r["probabilities_close"] for r in comparisons),
            "atol": 1e-4,
            "rtol": 1e-3,
            "max_abs_error": max(r["max_probability_abs_error"] for r in comparisons),
            "note": "Probability agreement is reported separately from weight integrity. BF16 inference differed between the training and fresh processes despite exact restored tensors; the precise numerical cause is unresolved. This check does not establish bitwise inference reproducibility.",
        },
        "checkpoint_files": files,
        "environment": environment(),
    }
    destination = Path(report_path) if report_path else output / "reload_verification.json"
    write_json(destination, result)
    if not ok:
        raise AssertionError(
            f"Checkpoint integrity or validation labels differ; inspect {destination}"
        )
    return result


def summarize(output=ROOT / "results"):
    from .inference import MODEL_KEYS

    output = Path(output)
    reports = [read_json(output / f"{key}.json") for key in MODEL_KEYS]
    for field in ("contract_hash",):
        if len({r[field] for r in reports}) != 1:
            raise ValueError(f"Mismatched {field}")
    for field in (
        "batch_size",
        "test_sha256",
        "evaluated_row_ids_sha256",
        "validation_warmup_sha256",
        "cuda_visible_devices",
    ):
        if len({r["protocol"][field] for r in reports}) != 1:
            raise ValueError(f"Mismatched benchmark {field}")
    if not all(r["protocol"]["full_official_test"] and r["metrics"]["n"] == 3080 for r in reports):
        raise ValueError(
            "Final benchmark cards require all 3,080 official test rows for every model"
        )
    if len({r["environment"]["gpu"]["name"] for r in reports}) != 1:
        raise ValueError("Different GPU hardware")
    rows = []
    for r in reports:
        prediction_file = output / f"{r['key']}.predictions.jsonl"
        if file_digest(prediction_file) != r["predictions_sha256"]:
            raise ValueError("Prediction file changed after scoring")
        rows.append(
            {
                "model": r["key"],
                "regime": r["regime"],
                "accuracy": r["metrics"]["accuracy"],
                "macro_f1": r["metrics"]["macro_f1"],
                "median_ms": r["latency_ms"]["median"],
                "p95_ms": r["latency_ms"]["p95"],
                "peak_gpu_gib": r["memory"]["peak_allocated_bytes"] / 2**30,
                "invalid": r["metrics"]["invalid_predictions"],
            }
        )
    write_json(output / "comparison.json", rows)
    import pandas as pd

    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    lines = [
        "# FeedbackLens benchmark cards",
        "",
        "All 3,080 official test messages; batch 1; BF16, no quantization. Latencies include preprocessing and output processing. GPU co-tenancy is recorded in the raw reports.",
        "",
        "| Model | Regime | Accuracy | Macro-F1 | Median ms | p95 ms | Peak GPU GiB | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['regime']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['peak_gpu_gib']:.3f} | {r['invalid']} |"
        )
    lines.extend(
        [
            "",
            "## Measurement conditions",
            "",
            "All four runs use the same physical RTX 4090, batch 1, 10 identical validation warmups, and the same ordered 77-intent contract. CUDA synchronization bounds tokenization, inference, decoding/parsing, and probability conversion. Model loading, warmup, disk writes, and scoring are excluded. Memory is the process's peak PyTorch allocated memory after warmup, including resident weights; reserved memory is also retained in each raw report.",
            "",
            "Other GPU workloads were present. Recorded latencies describe this shared-machine run and do not establish isolated-hardware speedups. Supervision differs: our head/LoRA uses BANKING77 train and validation data; the other models receive no local BANKING77 tuning. Unknown upstream training contamination is not audited.",
        ]
    )
    for report in reports:
        key = report["key"]
        packages = report["environment"]["packages"]
        lines.extend(
            [
                "",
                f"## {key}",
                "",
                f"- Regime: {report['regime']}.",
                f"- Base/model: `{report['reference']['repo_id']}` at `{report['reference']['revision']}`.",
                f"- Output: {report['constraint']}; thinking: {report['thinking']}.",
                f"- Parameter dtypes: {', '.join(report['parameter_dtypes'])}; quantization: {report['quantization']}.",
                f"- Runtime: torch {packages['torch']}; Transformers {packages['transformers']}; CUDA {report['environment']['cuda']}.",
                f"- Evidence: [raw report]({key}.json), [all row predictions and timings]({key}.predictions.jsonl).",
            ]
        )
    training = read_json(ROOT / "artifacts/decision/training_report.json")
    lines.extend(
        [
            "",
            "## Supervised training and checkpoint",
            "",
            f"Three epochs; best validation loss {training['best_validation_loss']:.4f}; validation-only calibration. Training took {training['training_seconds'] / 60:.2f} minutes with {training['peak_allocated_bytes'] / 2**30:.3f} GiB peak allocated memory. Timing includes epoch validation/checkpoint writes and excludes preparation, model loading, final calibration, and export.",
            "",
            "Fresh reload verifies exact head/adapter tensors, saved configuration, tokenizer probes, and eight validation labels. Confidence differences across the training and fresh BF16 processes are recorded separately; bitwise inference reproducibility is not established. See [verification](../artifacts/decision/reload_verification.json) and [training report](../artifacts/decision/training_report.json).",
        ]
    )
    (output / "BENCHMARKS.md").write_text("\n".join(lines) + "\n")
    return rows
