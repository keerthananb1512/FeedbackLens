from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from feedbacklens.client import ModelClient
from feedbacklens.common import ROOT, read_json
from feedbacklens.demo import messages_from_csv, prediction_frame
from feedbacklens.schema import load_schema

st.set_page_config(page_title="FeedbackLens", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background: #f7f8fb;} .block-container {max-width: 1240px; padding-top: 2.5rem;}
h1 {letter-spacing: -0.05em;} [data-testid="stMetric"] {background: white; padding: 1rem 1.2rem; border: 1px solid #e5e7ef; border-radius: 12px;}
</style>""",
    unsafe_allow_html=True,
)
st.caption("BANKING INTELLIGENCE / 77 INTENTS")
st.title("FeedbackLens")
st.write(
    "Turn customer messages into clear intent. Explore the accuracy, speed, and memory costs of generation and decision models."
)

MODELS = {
    "Our Qwen decision model · supervised": "decision",
    "Qwen structured JSON · zero-shot": "qwen_json",
    "LiquidAI d1-2B · zero-shot": "d1",
    "d3-nano · zero-shot": "d3",
}


@st.cache_resource
def model_client():
    return ModelClient()


analyze, compare, explore = st.tabs(["Classify messages", "Model comparison", "Intent catalog"])
with analyze:
    left, right = st.columns([2, 1])
    with left:
        selected = st.selectbox("Model", list(MODELS))
        mode = st.radio("Input", ["Paste messages", "Upload CSV"], horizontal=True)
        messages = []
        source = None
        if mode == "Paste messages":
            pasted = st.text_area(
                "One customer message per line",
                value="My card has not arrived yet.\nWhy was I charged twice for the same purchase?",
                height=170,
            )
            messages = [line.strip() for line in pasted.splitlines() if line.strip()]
        else:
            upload = st.file_uploader("CSV with a text column", type=["csv"])
            st.download_button(
                "Download sample CSV",
                'text\n"My card has not arrived yet."\n"How do I change my PIN?"\n',
                "feedbacklens-sample.csv",
                "text/csv",
            )
            if upload:
                try:
                    source, messages = messages_from_csv(upload.getvalue())
                    st.caption(f"{len(messages):,} messages ready")
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(str(exc))
        if len(messages) > 1000:
            st.error("Upload at most 1,000 messages per request.")
        if st.button(
            "Classify messages", type="primary", disabled=not messages or len(messages) > 1000
        ):
            key = MODELS[selected]
            if (
                key == "decision"
                and not (ROOT / "artifacts/decision/best/feedbacklens.json").exists()
            ):
                st.error(
                    "The trained checkpoint is not available yet. Training status is in logs/train.log."
                )
            else:
                try:
                    with st.spinner("Loading model and classifying messages…"):
                        predictions = model_client().predict(key, messages)
                    st.session_state["analysis"] = {
                        "messages": messages,
                        "predictions": predictions,
                        "model": selected,
                        "source": source.to_dict("records") if source is not None else None,
                    }
                except (RuntimeError, ValueError, TimeoutError) as exc:
                    st.error(str(exc))
    with right:
        st.subheader("One decision, 77 possibilities")
        st.write(
            "Each model sees the same intent descriptions and option order. Decision models return a probability for every intent in one call. The JSON model generates a valid intent label."
        )
        st.caption(
            "Probabilities are model estimates. They may be overconfident, especially on messages outside BANKING77."
        )
    if "analysis" in st.session_state:
        result = st.session_state["analysis"]
        st.subheader("Predictions")
        st.caption(result["model"])
        frame = prediction_frame(result["messages"], result["predictions"])
        st.dataframe(frame.drop(columns=["probabilities_json"]), hide_index=True, width="stretch")
        st.caption(
            "Live timings can include first-call kernel warmup. The comparison tab uses warmed models."
        )
        export = frame.copy()
        if result["source"] is not None:
            original = pd.DataFrame(result["source"]).add_prefix("source_")
            export = pd.concat([original, export], axis=1)
        st.download_button(
            "Download predictions CSV",
            export.to_csv(index=False),
            "feedbacklens-predictions.csv",
            "text/csv",
        )
        st.download_button(
            "Download predictions and probabilities JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            "feedbacklens-predictions.json",
            "application/json",
        )
        if any(p.get("probabilities") for p in result["predictions"]):
            index = st.selectbox(
                "Inspect probabilities for message",
                range(len(result["messages"])),
                format_func=lambda i: f"{i + 1}. {result['messages'][i][:90]}",
            )
            probabilities = result["predictions"][index].get("probabilities")
            if probabilities:
                values = pd.DataFrame(
                    {"intent": list(probabilities), "probability": list(probabilities.values())}
                ).sort_values("probability", ascending=False)
                st.bar_chart(values.head(8).set_index("intent"), horizontal=True, color="#5956d6")
                with st.expander("All 77 intent probabilities"):
                    st.dataframe(values, hide_index=True, width="stretch")

with compare:
    comparison = ROOT / "results/comparison.json"
    if comparison.exists():
        rows = read_json(comparison)
        frame = pd.DataFrame(rows)
        ours = frame[frame.model == "decision"].iloc[0]
        cols = st.columns(4)
        for col, label, value in zip(
            cols,
            ["Our accuracy", "Our macro-F1", "Median latency", "Peak GPU memory"],
            [
                f"{ours.accuracy:.1%}",
                f"{ours.macro_f1:.3f}",
                f"{ours.median_ms:.1f} ms",
                f"{ours.peak_gpu_gib:.2f} GiB",
            ],
        ):
            col.metric(label, value)
        st.dataframe(frame, hide_index=True, width="stretch")
        chart1, chart2 = st.columns(2)
        chart1.subheader("Accuracy and macro-F1")
        chart1.bar_chart(
            frame.set_index("model")[["accuracy", "macro_f1"]],
            stack=False,
            y_label="Score (0–1)",
        )
        chart2.subheader("Median and p95 latency")
        chart2.bar_chart(
            frame.set_index("model")[["median_ms", "p95_ms"]],
            stack=False,
            y_label="Milliseconds",
        )
        st.subheader("Peak inference GPU memory")
        st.bar_chart(frame.set_index("model")[["peak_gpu_gib"]], y_label="GiB")
        st.caption(
            "3,080 official test messages · same GPU · batch 1 · BF16 without quantization · 10 warmups · tokenization and output processing included. Model loading is excluded. GPU contention is recorded in the reports; these are local measurements, not isolated hardware claims."
        )
        training = ROOT / "artifacts/decision/training_report.json"
        if training.exists():
            report = read_json(training)
            st.write(
                f"Training: {report['training_seconds'] / 60:.1f} minutes; peak allocated GPU memory {report['peak_allocated_bytes'] / 2**30:.2f} GiB."
            )
        st.download_button(
            "Download benchmark table",
            frame.to_csv(index=False),
            "feedbacklens-benchmark.csv",
            "text/csv",
        )
        with st.expander("Measurement protocol and model details"):
            key = st.selectbox("Benchmark report", list(MODELS.values()))
            st.json(read_json(ROOT / f"results/{key}.json"))
    else:
        st.info(
            "Measured benchmark results appear here after all four full test runs finish. No estimated scores are shown."
        )
with explore:
    st.write("The shared catalog preserves the official BANKING77 label IDs and ordering.")
    st.dataframe(pd.DataFrame(load_schema()), hide_index=True, width="stretch")
