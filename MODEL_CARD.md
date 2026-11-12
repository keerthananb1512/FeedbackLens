---
language:
  - en
license: apache-2.0
base_model: unsloth/Qwen3.5-0.8B
base_model_relation: adapter
library_name: unsloth
pipeline_tag: text-classification
datasets:
  - PolyAI/banking77
tags:
  - banking77
  - intent-classification
  - finance
  - lora
  - peft
  - clef
  - qwen3_5
inference: false
model-index:
  - name: FeedbackLens-Qwen3.5-0.8B-BANKING77
    results:
      - task:
          type: text-classification
          name: Intent classification
        dataset:
          type: PolyAI/banking77
          name: BANKING77
          config: default
          split: test
          revision: 1846984189f2bc0ac51c6fc3a600fa9e1714e0c6
        metrics:
          - type: accuracy
            name: Accuracy
            value: 0.924025974025974
          - type: f1
            name: Macro-F1
            value: 0.9237919189536611
            args:
              average: macro
        source:
          name: FeedbackLens full official-test evaluation
          url: https://github.com/keerthananb1512/FeedbackLens/blob/329d98161a1970b1c40e3174542b63d66b3c5a04/results/decision.json
---

# FeedbackLens: BANKING77 Clef + LoRA

FeedbackLens classifies English banking messages into **77 intents** using a **Qwen3.5-0.8B backbone, a newly trained Clef decision head, and LoRA adapters**. It answers one multiple-choice question over the full intent catalog and returns a label with probabilities for all 77 options.

On all **3,080 official BANKING77 test messages**, it achieved **92.40% accuracy** and **0.9238 macro-F1**. This is a supervised BANKING77 model; the comparison models below received no local BANKING77 fine-tuning.

**Developer:** [keerthananb1512](https://github.com/keerthananb1512) · **Shared by:** [Keerthunb](https://huggingface.co/Keerthunb) · **Source and Streamlit demo:** [FeedbackLens](https://github.com/keerthananb1512/FeedbackLens) · **Detailed results:** [benchmark cards](https://github.com/keerthananb1512/FeedbackLens/blob/329d98161a1970b1c40e3174542b63d66b3c5a04/results/BENCHMARKS.md)

## Model and checkpoint

| Property | Configuration |
|---|---|
| Base | `unsloth/Qwen3.5-0.8B`, revision `23c69c53358a07516b5827588b3fdb12ae78fd65` |
| Architecture | Text-based Clef decision model with one 77-option choice field |
| Clef head | Width 512; 2 evidence-routing layers; 4 decoder layers; 8 attention heads; FFN width 2,048 |
| LoRA | Rank 16, alpha 16, dropout 0; language-model attention and feed-forward projections |
| Parameters | 891,133,508 total in the training wrapper; 38,147,588 trainable, approximately 4.28% |
| Precision | BF16 backbone with FP32 head/adapters; no quantization |
| Sequence length | 4,096 tokens during training |

The model release contains the **LoRA adapter, exact FP32 Clef head, tokenizer/processor, calibration configuration, and ordered intent schema**. The base model weights are a separate dependency. Loading uses the project's `Backend`, which reconstructs the model with Unsloth's `FastDecisionModel` and the pinned base revision.

The implementation adapts [Unsloth's official decision notebook](https://github.com/unslothai/notebooks/blob/59d5a1b8216e5f67c5e3ccf37a5ecae94a950a14/nb/Qwen3_5_%284B%29-Decision.ipynb). The model is trained for intent classification; its generative or multimodal capabilities were not evaluated.

## Get started

Use Linux, Python 3.11, an NVIDIA GPU supporting BF16, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The setup script installs the tested runtime, including Unsloth 2026.11.3, Transformers 5.17.0, PEFT 0.21.2, torch 2.9.0, and CUDA 12.8 wheels.

```bash
git clone https://github.com/keerthananb1512/FeedbackLens.git
cd FeedbackLens
bash scripts/setup.sh
```

The following example downloads the trained adapter/head release and the pinned base weights. Save it as `predict_example.py` and run it with `CUDA_VISIBLE_DEVICES=0 .venv/bin/python predict_example.py`:

```python
from huggingface_hub import snapshot_download
from feedbacklens.inference import Backend

# The loader resolves the pinned base from the local Hub cache.
snapshot_download(
    "unsloth/Qwen3.5-0.8B",
    revision="23c69c53358a07516b5827588b3fdb12ae78fd65",
)
checkpoint = snapshot_download(
    "Keerthunb/FeedbackLens-Qwen3.5-0.8B-BANKING77"
)
backend = Backend("decision", checkpoint=checkpoint)
prediction = backend.predict("My card has not arrived yet.")
print(prediction.label)
print(prediction.probabilities)  # All 77 intent probabilities.
```

For the Streamlit demo, use `snapshot_download("Keerthunb/FeedbackLens-Qwen3.5-0.8B-BANKING77", local_dir="artifacts/decision/best")` to place the trained checkpoint in the expected directory, then run `FEEDBACKLENS_GPU=0 .venv/bin/streamlit run app.py`. The app supports pasted messages, CSV uploads, probability inspection, prediction downloads, and the measured comparison dashboard.

## Training data and procedure

Data come from [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77), revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`. The official test split remains unchanged, including its internal duplicates.

From the original 10,003 training rows, preparation removes 25 rows overlapping the test set, 30 normalized duplicates, and 2 conflicting-label rows. Normalization uses Unicode NFKC, case folding, word tokens, and collapsed whitespace. A stratified split with seed 3407 produces **7,956 training / 1,990 validation** messages. All splits retain all 77 labels, with no normalized exact overlap between splits. The [data audit](https://github.com/keerthananb1512/FeedbackLens/blob/329d98161a1970b1c40e3174542b63d66b3c5a04/data/audit.json) records counts and hashes.

Training uses three epochs, physical batch size 2, gradient accumulation 16, effective batch size 32, AdamW, cosine scheduling, 5% warmup, weight decay 0.01, and seed 3407. The LoRA learning rate is `2e-4`; the head learning rate is `1e-4`. The best validation-loss checkpoint is selected, followed by validation-only temperature calibration. Training never loads the test messages.

The recorded run took **130.48 minutes** and **2.776 GiB peak allocated GPU memory** on one RTX 4090. Training timing includes epoch validation and checkpoint writes; it excludes model loading, data preparation, final calibration, and export. Energy use and carbon emissions were not measured.

## Evaluation

| Model | BANKING77 regime | Accuracy | Macro-F1 | Median latency | p95 latency | Peak GPU memory |
|---|---|---:|---:|---:|---:|---:|
| **FeedbackLens Clef + LoRA** | **Supervised** | **92.40%** | **0.9238** | **108.43 ms** | **114.26 ms** | **1.890 GiB** |
| Qwen3.5-0.8B structured JSON | Zero-shot | 61.69% | 0.6045 | 339.13 ms | 486.57 ms | 1.711 GiB |
| LiquidAI d1-2B | Zero-shot | 81.33% | 0.8062 | 90.27 ms | 94.59 ms | 3.650 GiB |
| d3-nano | Zero-shot | 87.82% | 0.8728 | 84.77 ms | 88.02 ms | 4.238 GiB |

Every model receives the same 3,080 test messages, intent descriptions, question, and option order. Runs use the same physical RTX 4090, batch size 1, 10 identical validation-message warmups, BF16 backbones, and no quantization. Qwen generation disables thinking and constrains complete JSON outputs to canonical labels; decision models use 77-option choice heads. Invalid predictions count as incorrect; all four recorded runs had zero invalid predictions.

CUDA synchronization bounds end-to-end latency, including tokenization, inference/generation, decoding/parsing, and probability conversion. Model loading, warmup, disk writes, and scoring are excluded. Memory is peak PyTorch allocated memory after warmup, including resident model weights; it excludes driver allocations and other processes.

**Other workloads shared the GPU.** These are local measurements, not isolated-hardware speed claims. FeedbackLens had the highest accuracy and used less GPU memory than d1/d3; those models had lower measured latency. Accuracy also reflects the difference in local BANKING77 supervision, so these results do not isolate the effect of architecture or model size.

## Intended use and limitations

- Suitable for banking-intent classification experiments, support-message routing prototypes, and studying decision-head versus structured-generation trade-offs.
- This is a closed-set classifier: unrelated, ambiguous, or unsupported messages still receive one of the 77 labels. No rejection threshold or out-of-domain detector was evaluated.
- Validation calibration does not guarantee reliable confidence on other domains. Non-English messages, real production traffic, demographic subgroups, and adversarial inputs were not evaluated.
- Fresh-process reloads restored every head/adapter tensor exactly and matched all eight validation-probe labels. Confidence values differed from the training process by up to **0.014** in the primary check and **0.007** in the independent-runtime check. The numerical cause remains unresolved; bitwise BF16 inference reproducibility is not established. [Verification evidence](https://github.com/keerthananb1512/FeedbackLens/blob/329d98161a1970b1c40e3174542b63d66b3c5a04/artifacts/decision/reload_verification.json).
- Duplicate removal covers normalized exact matches. Semantic paraphrase leakage and unknown upstream training-data contamination are not audited.
- Production throughput, concurrent serving capacity, and deployment cost were not measured. The model predicts intent; it does not authorize transactions or provide financial advice.

## License and attribution

Released weights/adapters use **Apache-2.0**, consistent with the [base model's license](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE). BANKING77 is separately licensed under **CC BY 4.0**. Please credit Iñigo Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), [paper](https://arxiv.org/abs/2003.04807), and the [PolyAI dataset](https://huggingface.co/datasets/PolyAI/banking77).
