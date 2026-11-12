from __future__ import annotations

from collections import Counter, defaultdict
import re
import unicodedata
from pathlib import Path

from .common import ROOT, digest, write_json, write_jsonl, file_digest


def normalized_text(text):
    # Case, Unicode compatibility, punctuation and whitespace variants form one group.
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def split_data(train, test, validation_fraction=0.2, seed=3407):
    """Keep official test intact; quarantine conflicting train groups and test overlaps."""
    from sklearn.model_selection import train_test_split

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = defaultdict(list)
    for row in train:
        if not normalized_text(row["text"]):
            raise ValueError("Empty normalized training message")
        groups[normalized_text(row["text"])].append(row)
    test_keys = {normalized_text(r["text"]) for r in test}
    clean, removed = [], []
    for key, rows in groups.items():
        reason = (
            "test_overlap"
            if key in test_keys
            else "conflicting_labels"
            if len({r["label"] for r in rows}) > 1
            else None
        )
        if reason:
            removed.extend({**r, "reason": reason} for r in rows)
        else:
            clean.append(rows[0])
            removed.extend({**r, "reason": "duplicate_train"} for r in rows[1:])
    fitting, validation = train_test_split(
        clean,
        test_size=validation_fraction,
        random_state=seed,
        stratify=[r["label"] for r in clean],
    )
    fitting.sort(key=lambda r: r["id"])
    validation.sort(key=lambda r: r["id"])
    sets = [{normalized_text(r["text"]) for r in rows} for rows in (fitting, validation, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return fitting, validation, list(test), removed


def prepare(output, revision=None, validation_fraction=0.2, seed=3407):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = revision or HfApi().dataset_info("PolyAI/banking77").sha
    dataset = load_dataset("PolyAI/banking77", revision=revision)
    labels = dataset["train"].features["label"].names
    schema_path = ROOT / "configs/intents.json"
    if not schema_path.exists():
        raise FileNotFoundError("Commit intent descriptions before preparing data")
    from .schema import load_schema

    schema = load_schema(schema_path)
    assert [r["label"] for r in schema] == labels, "Schema differs from official label order"
    raw = {
        s: [
            {"id": f"{s}:{i}", "text": r["text"], "label": r["label"]}
            for i, r in enumerate(dataset[s])
        ]
        for s in ("train", "test")
    }
    train, validation, test, removed = split_data(
        raw["train"], raw["test"], validation_fraction, seed
    )
    output = Path(output)
    for name, rows in (
        ("train", train),
        ("validation", validation),
        ("test", test),
        ("removed", removed),
    ):
        write_jsonl(output / f"{name}.jsonl", rows)
    assert test == raw["test"]
    counts = Counter(normalized_text(r["text"]) for r in test)
    audit = {
        "dataset": "PolyAI/banking77",
        "revision": revision,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "normalization": "NFKC + casefold + Unicode word tokens + collapsed whitespace",
        "official_train_rows": len(raw["train"]),
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "removed": dict(Counter(r["reason"] for r in removed)),
        "cross_split_normalized_overlap": 0,
        "official_test_preserved": test == raw["test"],
        "official_test_sha256": digest(raw["test"]),
        "test_duplicate_groups": sum(v > 1 for v in counts.values()),
        "per_class_counts": {
            name: dict(Counter(r["label"] for r in rows))
            for name, rows in (("train", train), ("validation", validation), ("test", test))
        },
        "files": {
            name: file_digest(output / f"{name}.jsonl")
            for name in ("train", "validation", "test", "removed")
        },
    }
    write_json(output / "audit.json", audit)
    return audit
