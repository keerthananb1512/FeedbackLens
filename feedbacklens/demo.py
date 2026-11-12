"""Input/output helpers shared by the Streamlit app and its tests."""

from __future__ import annotations

import io
import json
import pandas as pd


def messages_from_csv(content: bytes, column="text"):
    frame = pd.read_csv(io.BytesIO(content), keep_default_na=False, dtype=str)
    if column not in frame.columns:
        raise ValueError(f"CSV must contain a {column!r} column")
    if frame.empty:
        raise ValueError("CSV has no messages")
    messages = frame[column].tolist()
    if any(not isinstance(s, str) or not s.strip() for s in messages):
        raise ValueError("Every CSV message must contain text")
    return frame, messages


def prediction_frame(messages, predictions):
    if len(messages) != len(predictions):
        raise ValueError("Prediction count differs from input count")
    return pd.DataFrame(
        [
            {
                "text": text,
                "predicted_intent": p["label"],
                "top_probability": max(p["probabilities"].values())
                if p.get("probabilities")
                else None,
                "latency_ms": p.get("latency_ms"),
                "error": p.get("error"),
                "probabilities_json": json.dumps(p["probabilities"])
                if p.get("probabilities")
                else "",
            }
            for text, p in zip(messages, predictions)
        ]
    )
