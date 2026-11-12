from collections import Counter

from feedbacklens.common import ROOT, digest, file_digest, read_json, read_jsonl
from feedbacklens.prepare import normalized_text, split_data


def test_duplicate_groups_are_removed_without_changing_test():
    train = [{"id": f"train:{i}", "text": f"Unique message {i}", "label": i % 2} for i in range(30)]
    train += [
        {"id": "train:dup", "text": "UNIQUE  message 0!!!", "label": 0},
        {"id": "train:overlap", "text": "TEST, MESSAGE", "label": 0},
        {"id": "train:conflict1", "text": "conflicting message", "label": 0},
        {"id": "train:conflict2", "text": "Conflicting Message!", "label": 1},
    ]
    test = [
        {"id": "test:0", "text": "test message", "label": 1},
        {"id": "test:1", "text": "test message", "label": 1},
    ]
    fit, val, result, removed = split_data(train, test)
    assert result == test
    assert Counter(r["reason"] for r in removed) == {
        "duplicate_train": 1,
        "test_overlap": 1,
        "conflicting_labels": 2,
    }
    keys = [{normalized_text(r["text"]) for r in rows} for rows in (fit, val, result)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert {r["label"] for r in val} == {0, 1}
    assert split_data(train, test) == (fit, val, result, removed)


def test_prepared_real_data_matches_audit_and_official_test():
    audit = read_json(ROOT / "data/audit.json")
    rows = {
        split: read_jsonl(ROOT / f"data/{split}.jsonl") for split in ("train", "validation", "test")
    }
    for split, items in rows.items():
        assert len(items) == audit["counts"][split]
        assert file_digest(ROOT / f"data/{split}.jsonl") == audit["files"][split]
        assert {r["label"] for r in items} == set(range(77))
    assert len(rows["test"]) == 3080
    assert set(Counter(r["label"] for r in rows["test"]).values()) == {40}
    assert digest(rows["test"]) == audit["official_test_sha256"]
    keys = [
        {normalized_text(r["text"]) for r in rows[split]}
        for split in ("train", "validation", "test")
    ]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
