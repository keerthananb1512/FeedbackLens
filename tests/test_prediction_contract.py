import json
import pytest

from feedbacklens.evaluate import classification_metrics
from feedbacklens.inference import LabelTrie, decision_prediction
from feedbacklens.schema import (
    contract_hash,
    generation_messages,
    load_schema,
    parse_generation,
    questions,
)


def test_intent_contract_preserves_77_labels_and_descriptions():
    schema = load_schema()
    criteria = questions(schema)["intent"]["criteria"]
    prompt = generation_messages("Customer text", schema)
    assert list(criteria) == [r["label"] for r in schema]
    for row in schema:
        assert f"{row['label']}: {row['description']}" in prompt[0]["content"]
    assert prompt[1]["content"] == "Customer text"
    assert contract_hash(schema) != contract_hash(list(reversed(schema)))


@pytest.mark.parametrize(
    "raw",
    [
        "bad output",
        '```json\n{"intent":"activate_my_card"}\n```',
        '{"intent":null}',
        '{"intent":"unknown"}',
        '{"intent":["activate_my_card"]}',
        '{"intent":"activate_my_card","reason":"because"}',
        "[]",
    ],
)
def test_invalid_generation_is_not_coerced(raw):
    assert parse_generation(raw, ["activate_my_card"]) is None


def test_valid_json_and_trie_allow_every_label_and_no_prose():
    class CharTokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(x) for x in text]

    labels = [r["label"] for r in load_schema()]
    trie = LabelTrie(CharTokenizer(), labels)
    for label in labels:
        raw = json.dumps({"intent": label}, separators=(",", ":"))
        assert parse_generation(raw, labels) == label
        prefix = []
        for token in [ord(x) for x in raw] + [0]:
            assert token in trie.allowed(prefix)
            prefix.append(token)
    assert ord("H") not in trie.allowed([])


def test_invalid_predictions_reduce_accuracy_and_macro_f1():
    metrics = classification_metrics(["a", "b", "c"], ["a", None, "wrong"], ["a", "b", "c"])
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["invalid_predictions"] == 2
    assert metrics["per_class"]["b"]["recall"] == 0
    assert metrics["confusion_matrix"][1][-1] == 1


def test_probabilities_require_complete_finite_distribution():
    assert (
        decision_prediction(
            {"choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}, ["a", "b"]
        ).label
        == "b"
    )
    with pytest.raises(ValueError):
        decision_prediction({"choice": "b", "probabilities": {"b": 0.9}}, ["a", "b"])
    with pytest.raises(ValueError):
        decision_prediction(
            {"choice": "b", "probabilities": {"a": float("nan"), "b": 0.9}}, ["a", "b"]
        )
