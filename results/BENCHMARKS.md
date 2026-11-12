# FeedbackLens benchmark cards

All 3,080 official test messages; batch 1; BF16, no quantization. Latencies include preprocessing and output processing. GPU co-tenancy is recorded in the raw reports.

| Model | Regime | Accuracy | Macro-F1 | Median ms | p95 ms | Peak GPU GiB | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| decision | supervised on BANKING77 train | 0.9240 | 0.9238 | 108.43 | 114.26 | 1.890 | 0 |
| qwen_json | zero-shot on BANKING77 (no local fine-tuning) | 0.6169 | 0.6045 | 339.13 | 486.57 | 1.711 | 0 |
| d1 | zero-shot on BANKING77 (no local fine-tuning) | 0.8133 | 0.8062 | 90.27 | 94.59 | 3.650 | 0 |
| d3 | zero-shot on BANKING77 (no local fine-tuning) | 0.8782 | 0.8728 | 84.77 | 88.02 | 4.238 | 0 |

## Measurement conditions

All four runs use the same physical RTX 4090, batch 1, 10 identical validation warmups, and the same ordered 77-intent contract. CUDA synchronization bounds tokenization, inference, decoding/parsing, and probability conversion. Model loading, warmup, disk writes, and scoring are excluded. Memory is the process's peak PyTorch allocated memory after warmup, including resident weights; reserved memory is also retained in each raw report.

Other GPU workloads were present. Recorded latencies describe this shared-machine run and do not establish isolated-hardware speedups. Supervision differs: our head/LoRA uses BANKING77 train and validation data; the other models receive no local BANKING77 tuning. Unknown upstream training contamination is not audited.

## decision

- Regime: supervised on BANKING77 train.
- Base/model: `unsloth/Qwen3.5-0.8B` at `23c69c53358a07516b5827588b3fdb12ae78fd65`.
- Output: 77-option choice head; thinking: not applicable.
- Parameter dtypes: torch.bfloat16, torch.float32; quantization: none.
- Runtime: torch 2.9.0+cu128; Transformers 5.17.0; CUDA 12.8.
- Evidence: [raw report](decision.json), [all row predictions and timings](decision.predictions.jsonl).

## qwen_json

- Regime: zero-shot on BANKING77 (no local fine-tuning).
- Base/model: `unsloth/Qwen3.5-0.8B` at `23c69c53358a07516b5827588b3fdb12ae78fd65`.
- Output: exact JSON label trie; thinking: False.
- Parameter dtypes: torch.bfloat16; quantization: none.
- Runtime: torch 2.9.0+cu128; Transformers 5.17.0; CUDA 12.8.
- Evidence: [raw report](qwen_json.json), [all row predictions and timings](qwen_json.predictions.jsonl).

## d1

- Regime: zero-shot on BANKING77 (no local fine-tuning).
- Base/model: `LiquidAI/d1-2B` at `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`.
- Output: 77-option choice head; thinking: not applicable.
- Parameter dtypes: torch.bfloat16; quantization: none.
- Runtime: torch 2.9.0+cu128; Transformers 5.19.0; CUDA 12.8.
- Evidence: [raw report](d1.json), [all row predictions and timings](d1.predictions.jsonl).

## d3

- Regime: zero-shot on BANKING77 (no local fine-tuning).
- Base/model: `vllm-sr/d3-nano` at `8198c9f763b046605b2f4827b4999971fa2c92c4`.
- Output: 77-option choice head; thinking: not applicable.
- Parameter dtypes: torch.bfloat16; quantization: none.
- Runtime: torch 2.9.0+cu128; Transformers 5.17.0; CUDA 12.8.
- Evidence: [raw report](d3.json), [all row predictions and timings](d3.predictions.jsonl).

## Supervised training and checkpoint

Three epochs; best validation loss 0.2791; validation-only calibration. Training took 130.48 minutes with 2.776 GiB peak allocated memory. Timing includes epoch validation/checkpoint writes and excludes preparation, model loading, final calibration, and export.

Fresh reload verifies exact head/adapter tensors, saved configuration, tokenizer probes, and eight validation labels. Confidence differences across the training and fresh BF16 processes are recorded separately; bitwise inference reproducibility is not established. See [verification](../artifacts/decision/reload_verification.json) and [training report](../artifacts/decision/training_report.json).
