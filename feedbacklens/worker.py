"""One persistent model per worker, with a JSON-lines protocol on stdout."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    protocol = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .inference import Backend

            backend = Backend(args.model, checkpoint=args.checkpoint)
        protocol.write(json.dumps({"ready": True, "metadata": backend.metadata}) + "\n")
        protocol.flush()
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("action") == "close":
                break
            results = []
            for message in request["messages"]:
                try:
                    if backend.torch.cuda.is_available():
                        backend.torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    with contextlib.redirect_stdout(sys.stderr):
                        result = backend.predict(message).to_dict()
                    if backend.torch.cuda.is_available():
                        backend.torch.cuda.synchronize()
                    result["latency_ms"] = (time.perf_counter_ns() - started) / 1e6
                except Exception as exc:
                    result = {"label": None, "probabilities": None, "error": str(exc)}
                results.append(result)
            protocol.write(json.dumps({"predictions": results}) + "\n")
            protocol.flush()
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        protocol.write(json.dumps({"ready": False, "error": str(exc)}) + "\n")
        protocol.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
