from __future__ import annotations

from pathlib import Path
import time

from .common import ROOT, environment, file_digest, read_json, read_jsonl, write_json
from .schema import contract_hash, load_schema, questions


def train(
    config_path=ROOT / "configs/train.json",
    data_dir=ROOT / "data",
    output=ROOT / "artifacts/decision",
    resume=None,
):
    from unsloth import FastDecisionModel, DecisionTrainer
    import torch
    from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

    config = read_json(config_path)
    output, data_dir = Path(output), Path(data_dir)
    if (output / "training_report.json").exists() and not resume:
        raise FileExistsError("Completed training exists; choose a new output directory")
    schema = load_schema()
    ref = read_json(ROOT / "configs/models.json")["qwen_json"]
    set_seed(config["seed"])
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=ref["repo_id"],
        revision=ref["revision"],
        decision_head="clef",
        max_seq_length=config["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
        device_map={"": "cuda:0"},
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )
    prepared, reports = {}, {}
    for split in ("train", "validation"):
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        typed = [
            {
                "state": r["text"],
                "questions": questions(schema),
                "gold": {"intent": schema[r["label"]]["label"]},
            }
            for r in rows
        ]
        prepared[split], reports[split] = FastDecisionModel.build_dataset(typed, tokenizer, model)
        if (
            reports[split]["skipped"]
            or reports[split]["truncated"]
            or len(prepared[split]) != len(rows)
        ):
            raise ValueError(f"Dataset was dropped or truncated: {reports[split]}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "dataset_encoding.json", reports)
    meta = {
        "config": config,
        "base_model": ref,
        "contract_hash": contract_hash(schema),
        "data_audit": read_json(data_dir / "audit.json"),
        "environment": environment(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "head_config": read_json(ROOT / "references/head_config.json")
        if (ROOT / "references/head_config.json").exists()
        else str(model.head),
        "source_notebook": "https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb",
        "notebook_sha256": file_digest(ROOT / "references/Qwen3_5_(4B)-Decision.ipynb"),
    }
    write_json(output / "run.json", meta)
    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        weight_decay=config["weight_decay"],
        bf16=True,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=config["seed"],
        dataloader_num_workers=0,
        prediction_loss_only=True,
        optim="adamw_torch",
    )
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=prepared["train"],
        eval_dataset=prepared["validation"],
        args=args,
        head_learning_rate=config["head_learning_rate"],
        permute_fields=False,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])
        ],
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    training_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    # Fit probabilities using validation only; no test messages are loaded in this module.
    calibration = FastDecisionModel.calibrate(model, tokenizer, prepared["validation"])
    FastDecisionModel.for_inference(model)
    probe_rows = read_jsonl(data_dir / "validation.jsonl")[:8]
    probes = [
        {
            "id": r["id"],
            "text": r["text"],
            "prediction": FastDecisionModel.predict(model, tokenizer, r["text"], questions(schema))[
                "intent"
            ],
        }
        for r in probe_rows
    ]
    # Public save_pretrained downcasts the head to BF16. Preserve the exact trained
    # FP32 head and its unfurled calibration using the pinned trainer's exporter.
    from unsloth.models.decision import _save_clef_adapter

    _save_clef_adapter(model, output / "best", tokenizer, exact=True)
    write_json(output / "best/feedbacklens.json", meta)
    write_json(output / "best/intents.json", schema)
    write_json(output / "reload_reference.json", probes)
    report = {
        **meta,
        **training_memory,
        "training_seconds": wall_seconds,
        "training_timing_scope": "trainer.train including validation and checkpoint saving; excludes loading, data preparation, final calibration and export",
        "trainer_metrics": result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
        "calibration": calibration,
        "export": str(output / "best"),
        "environment_after": environment(),
    }
    write_json(output / "training_report.json", report)
    return report
