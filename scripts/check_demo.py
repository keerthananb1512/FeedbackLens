"""Exercise the real Streamlit classify interaction with an installed model."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from streamlit.testing.v1 import AppTest
from feedbacklens.common import ROOT, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["qwen_json", "decision"], default="decision")
args = parser.parse_args()
choice = (
    "Our Qwen decision model · supervised"
    if args.model == "decision"
    else "Qwen structured JSON · zero-shot"
)
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
at.selectbox[0].select(choice).run()
at.button[0].click().run(timeout=240)
assert not at.exception, at.exception
result = at.session_state["analysis"]
assert len(result["predictions"]) == 2
assert all(p["label"] and not p.get("error") for p in result["predictions"])
if args.model == "decision":
    assert all(len(p["probabilities"]) == 77 for p in result["predictions"])
write_json(ROOT / f"results/app_interaction_{args.model}.json", {"passed": True, "result": result})
st.cache_resource.clear()
print("Real Streamlit classify interaction passed")
