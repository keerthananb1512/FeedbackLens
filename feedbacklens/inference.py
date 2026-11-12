from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .common import ROOT, read_json
from .schema import contract_hash, generation_messages, load_schema, parse_generation, questions

MODEL_KEYS = ("decision", "qwen_json", "d1", "d3")


@dataclass
class Prediction:
    label: str | None
    probabilities: dict[str, float] | None = None
    raw: str | dict | None = None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class LabelTrie:
    """Only exact JSON strings containing one of the 77 labels can be generated."""

    def __init__(self, tokenizer, labels):
        self.root = {}
        self.max_tokens = 0
        self.eos = tokenizer.eos_token_id
        for label in labels:
            ids = tokenizer.encode(
                json.dumps({"intent": label}, separators=(",", ":")), add_special_tokens=False
            )
            self.max_tokens = max(self.max_tokens, len(ids))
            node = self.root
            for token in ids + [self.eos]:
                node = node.setdefault(token, {})

    def allowed(self, suffix):
        node = self.root
        for token in suffix:
            if token not in node:
                raise ValueError("Generated prefix left the valid JSON trie")
            node = node[token]
        return list(node) or [self.eos]


def decision_prediction(answer, labels):
    import math

    label = answer.get("answer", answer.get("choice"))
    probabilities = answer.get("probabilities")
    if label not in labels:
        return Prediction(None, raw=answer, error="Invalid decision label")
    if probabilities is not None:
        if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
            raise ValueError("Decision probabilities must cover all 77 options")
        probabilities = {label: float(probabilities[label]) for label in labels}
        if (
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > 0.005
        ):
            raise ValueError("Invalid decision probability distribution")
    return Prediction(label, probabilities, raw=answer)


class Backend:
    def __init__(self, key, checkpoint=None, device="cuda:0", schema=None):
        if key not in MODEL_KEYS:
            raise ValueError(f"Unknown model: {key}")
        self.key, self.device = key, device
        self.schema = schema or load_schema()
        self.labels = [r["label"] for r in self.schema]
        self.questions = questions(self.schema)
        self.contract_hash = contract_hash(self.schema)
        self.ref = read_json(ROOT / "configs/models.json").get(key)
        import torch

        self.torch = torch
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        if key == "decision":
            # Import before transformers, as required by Unsloth's patch initialization.
            from unsloth import FastDecisionModel

            self.api = FastDecisionModel
            checkpoint = Path(checkpoint or ROOT / "artifacts/decision/best").resolve()
            meta = read_json(checkpoint / "feedbacklens.json")
            if meta["contract_hash"] != self.contract_hash:
                raise ValueError("Checkpoint intent descriptions or option order differ")
            self.ref = meta["base_model"]
            self.model, self.tokenizer = self.api.from_pretrained(
                str(checkpoint),
                dtype=self.dtype,
                load_in_4bit=False,
                max_seq_length=meta["config"]["max_seq_length"],
                use_gradient_checkpointing=False,
                device_map={"": device},
                local_files_only=True,
            )
            self.api.for_inference(self.model)
        elif key == "qwen_json":
            from transformers import AutoModelForImageTextToText, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.ref["repo_id"], revision=self.ref["revision"]
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.ref["repo_id"],
                revision=self.ref["revision"],
                dtype=self.dtype,
                attn_implementation="sdpa",
                device_map={"": device},
            ).eval()
            self.trie = LabelTrie(self.tokenizer, self.labels)
        else:
            from transformers import AutoModel

            self.model = (
                AutoModel.from_pretrained(
                    self.ref["repo_id"],
                    revision=self.ref["revision"],
                    code_revision=self.ref["revision"],
                    trust_remote_code=True,
                    dtype=self.dtype,
                )
                .to(device)
                .eval()
            )
        self.metadata = {
            "key": key,
            "reference": self.ref,
            "regime": "supervised on BANKING77 train"
            if key == "decision"
            else "zero-shot on BANKING77 (no local fine-tuning)",
            "precision": str(self.dtype),
            "quantization": "none",
            "contract_hash": self.contract_hash,
            "batch_size": 1,
            "thinking": False if key == "qwen_json" else "not applicable",
            "constraint": "exact JSON label trie"
            if key == "qwen_json"
            else "77-option choice head",
            "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()}),
        }

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message must be a nonempty string")
        with self.torch.inference_mode():
            if self.key == "decision":
                out = self.api.predict(self.model, self.tokenizer, text, self.questions)["intent"]
                return decision_prediction(out, self.labels)
            if self.key in ("d1", "d3"):
                out = self.model.system_one(state=text, questions=self.questions)["answers"][
                    "intent"
                ]
                return decision_prediction(out, self.labels)
            encoded = self.tokenizer.apply_chat_template(
                generation_messages(text, self.schema),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_length = encoded["input_ids"].shape[1]
            tokens = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.trie.max_tokens + 1,
                prefix_allowed_tokens_fn=lambda batch, ids: self.trie.allowed(
                    ids[prompt_length:].tolist()
                ),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                logits_to_keep=1,
            )
            raw = self.tokenizer.decode(tokens[0, prompt_length:], skip_special_tokens=True)
            label = parse_generation(raw, self.labels)
            return Prediction(
                label, raw=raw, error=None if label else "Invalid structured prediction"
            )
