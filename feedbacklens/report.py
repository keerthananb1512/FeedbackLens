from __future__ import annotations

from .common import ROOT, read_json


def write_tradeoff_plot(rows, destination=ROOT / "results/tradeoffs"):
    """Plot measured rows; marker area is proportional to peak allocated GPU memory."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = {
        "decision": "Our Clef + LoRA\n(supervised)",
        "qwen_json": "Qwen3.5-0.8B JSON\n(zero-shot)",
        "d1": "d1-2B (zero-shot)",
        "d3": "d3-nano (zero-shot)",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.22)
        for row in rows:
            ours = row["model"] == "decision"
            ax.scatter(
                row["median_ms"],
                row["accuracy"],
                s=180 * row["peak_gpu_gib"],
                color="#5956d6" if ours else "#287e9c",
                alpha=0.85,
                marker="o",
                edgecolors="white",
                linewidths=1.3,
                zorder=3,
            )
            offset = (
                (-12, -22)
                if row["model"] == "qwen_json"
                else (12, -22)
                if row["model"] == "d1"
                else (12, 12)
            )
            ax.annotate(
                names[row["model"]],
                (row["median_ms"], row["accuracy"]),
                xytext=offset,
                textcoords="offset points",
                ha="right" if row["model"] == "qwen_json" else "left",
                fontsize=9,
                color="#18223a",
            )
        ax.set(
            xlabel="Median end-to-end latency (ms)",
            ylabel="Official test accuracy",
            xlim=(0, max(r["median_ms"] for r in rows) * 1.22),
            ylim=(
                max(0, min(r["accuracy"] for r in rows) - 0.09),
                min(1.04, max(r["accuracy"] for r in rows) + 0.09),
            ),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.18, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            "FeedbackLens: accuracy, latency, and GPU memory", loc="left", pad=22, fontweight="bold"
        )
        for size in (2, 4):
            ax.scatter([], [], s=180 * size, color="#82909f", alpha=0.65, label=f"{size} GiB")
        ax.legend(
            title="Peak allocated GPU memory",
            loc="lower left",
            frameon=False,
            labelspacing=1.5,
            borderpad=1,
        )
        fig.text(
            0.02,
            0.015,
            "3,080 official test messages · RTX 4090 · batch 1 · BF16 · shared GPU\nAccuracy also reflects BANKING77 supervision; model loading is excluded from latency.",
            fontsize=9,
            color="#526078",
        )
        for extension in ("png", "svg"):
            fig.savefig(
                destination.with_suffix(f".{extension}"),
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)


def update_readme():
    rows = read_json(ROOT / "results/comparison.json")
    training = read_json(ROOT / "artifacts/decision/training_report.json")
    verification = read_json(ROOT / "artifacts/decision/reload_verification.json")
    if not verification["passed"]:
        raise ValueError("Checkpoint verification must pass before publishing results")
    write_tradeoff_plot(rows)
    lines = [
        "Measured on the full official test split. See [benchmark cards](results/BENCHMARKS.md) and raw reports for the protocol, software versions, and GPU contention records.",
        "",
        "| Model | BANKING77 regime | Accuracy | Macro-F1 | Median / p95 ms | Peak GPU GiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    names = {
        "decision": "Our Qwen Clef + LoRA",
        "qwen_json": "Qwen3.5-0.8B JSON",
        "d1": "LiquidAI d1-2B",
        "d3": "d3-nano",
    }
    for row in rows:
        regime = "supervised" if row["model"] == "decision" else "zero-shot"
        lines.append(
            f"| {names[row['model']]} | {regime} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | "
            f"{row['median_ms']:.2f} / {row['p95_ms']:.2f} | {row['peak_gpu_gib']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Training: **{training['training_seconds'] / 60:.2f} min**, **{training['peak_allocated_bytes'] / 2**30:.3f} GiB** peak allocated GPU memory. Fresh checkpoint reload restored every head/adapter tensor exactly and matched all {len(verification['comparisons'])} validation labels. Probability differences are recorded separately in the verification report; BF16 inference is not bitwise reproducible across these processes.",
            "",
            "![Accuracy versus latency, with marker area showing GPU memory](results/tradeoffs.png)",
        ]
    )
    path = ROOT / "README.md"
    text = path.read_text()
    start, end = "<!-- RESULTS:start -->", "<!-- RESULTS:end -->"
    before, rest = text.split(start)
    _, after = rest.split(end)
    path.write_text(before + start + "\n" + "\n".join(lines) + "\n" + end + after)
