"""Exercise CSV upload, real inference, exports, and layout through a browser."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8508")
    parser.add_argument("--model", choices=["d1", "decision", "qwen_json"], default="decision")
    parser.add_argument("--chromium")
    args = parser.parse_args()
    output = ROOT / "results/browser"
    output.mkdir(exist_ok=True, parents=True)
    choice = {
        "d1": "LiquidAI d1-2B · zero-shot",
        "decision": "Our Qwen decision model · supervised",
        "qwen_json": "Qwen structured JSON · zero-shot",
    }[args.model]
    launch = {"headless": True}
    if args.chromium:
        launch["executable_path"] = args.chromium
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.get_by_role("heading", name="FeedbackLens", exact=True)).to_be_visible(
            timeout=30000
        )
        expect(page.get_by_role("button", name="Classify messages", exact=True)).to_be_enabled()
        page.screenshot(path=str(output / "initial.png"), full_page=True)
        page.get_by_role("combobox").first.click()
        page.get_by_role("option", name=choice, exact=True).click()
        page.get_by_text("Upload CSV", exact=True).click()
        page.locator("input[type=file]").set_input_files(
            {
                "name": "messages.csv",
                "mimeType": "text/csv",
                "buffer": b'id,text\n001,"My card has not arrived yet."\n002,"How do I change my PIN?"\n',
            }
        )
        expect(page.get_by_text("2 messages ready", exact=True)).to_be_visible()
        print("CSV uploaded and parsed", flush=True)
        page.get_by_role("button", name="Classify messages", exact=True).click()
        expect(page.get_by_role("heading", name="Predictions", exact=True)).to_be_visible(
            timeout=240000
        )
        print("Real model predictions displayed", flush=True)
        with page.expect_download() as info:
            page.get_by_role("button", name="Download predictions CSV", exact=True).click()
        info.value.save_as(output / f"{args.model}-predictions.csv")
        with page.expect_download() as info:
            page.get_by_role(
                "button", name="Download predictions and probabilities JSON", exact=True
            ).click()
        info.value.save_as(output / f"{args.model}-predictions.json")
        result = json.loads((output / f"{args.model}-predictions.json").read_text())
        rows = list(
            csv.DictReader(io.StringIO((output / f"{args.model}-predictions.csv").read_text()))
        )
        assert len(rows) == len(result["predictions"]) == 2
        assert rows[0]["source_id"] == "001"
        assert result["model"] == choice
        assert all(r["label"] and not r.get("error") for r in result["predictions"])
        if args.model != "qwen_json":
            assert all(len(r["probabilities"]) == 77 for r in result["predictions"])
            expect(page.get_by_text("All 77 intent probabilities", exact=True)).to_be_visible()
        assert not errors, errors
        page.screenshot(path=str(output / f"{args.model}-csv-results.png"), full_page=True)
        if args.model != "qwen_json":
            page.get_by_text("All 77 intent probabilities", exact=True).click()
            page.get_by_text("All 77 intent probabilities", exact=True).scroll_into_view_if_needed()
            page.screenshot(path=str(output / f"{args.model}-probabilities.png"))
        page.get_by_role("tab", name="Model comparison", exact=True).click()
        expect(page.get_by_text("Our accuracy", exact=True)).to_be_visible()
        with page.expect_download() as info:
            page.get_by_role("button", name="Download benchmark table", exact=True).click()
        info.value.save_as(output / "comparison.csv")
        comparison = list(csv.DictReader((output / "comparison.csv").open()))
        assert {r["model"] for r in comparison} == {"decision", "qwen_json", "d1", "d3"}
        expected_comparison = json.loads((ROOT / "results/comparison.json").read_text())
        for row in comparison:
            expected = next(r for r in expected_comparison if r["model"] == row["model"])
            assert float(row["accuracy"]) == expected["accuracy"]
            assert float(row["macro_f1"]) == expected["macro_f1"]
        page.get_by_role("heading", name="FeedbackLens", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "comparison.png"), full_page=True)
        page.get_by_text("Accuracy and macro-F1", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "comparison-charts.png"))
        page.get_by_text("Peak inference GPU memory", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "comparison-memory.png"))
        page.get_by_role("tab", name="Classify messages", exact=True).click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / f"{args.model}-mobile.png"), full_page=True)
        report = {
            "passed": True,
            "model": args.model,
            "browser": browser.version,
            "uploaded_rows": 2,
            "csv_download_rows": len(rows),
            "all_77_probabilities": args.model != "qwen_json",
            "benchmark_models_verified": len(comparison),
            "page_errors": errors,
        }
        (output / f"{args.model}-verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report)
        browser.close()


if __name__ == "__main__":
    main()
