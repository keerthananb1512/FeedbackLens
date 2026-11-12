from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading

from .common import ROOT


class ModelClient:
    """Serialize app requests and release a previous model before loading the next."""

    def __init__(self):
        self.process = None
        self.model_key = None
        self.lock = threading.RLock()
        self.log = None
        atexit.register(self.close)

    def _read(self, timeout=600):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("Model did not respond; inspect logs/demo-worker.log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Model worker stopped; inspect logs/demo-worker.log")
        value = json.loads(line)
        if value.get("error"):
            raise RuntimeError(value["error"])
        return value

    def predict(self, key, messages):
        with self.lock:
            if not self.process or self.process.poll() is not None or key != self.model_key:
                self.close()
                executable = ROOT / (".venv-d1/bin/python" if key == "d1" else ".venv/bin/python")
                if not executable.exists():
                    if key == "d1":
                        raise RuntimeError("Create .venv-d1 using scripts/setup.sh")
                    executable = Path(sys.executable)
                env = {
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": os.environ.get("FEEDBACKLENS_GPU", "1"),
                }
                (ROOT / "logs").mkdir(exist_ok=True)
                self.log = (ROOT / "logs/demo-worker.log").open("a")
                self.process = subprocess.Popen(
                    [str(executable), "-m", "feedbacklens.worker", "--model", key],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self.log,
                    text=True,
                    bufsize=1,
                )
                self.model_key = key
                try:
                    ready = self._read()
                    if not ready.get("ready"):
                        raise RuntimeError("Model worker did not initialize")
                except (RuntimeError, TimeoutError, ValueError):
                    self.close()
                    raise
            self.process.stdin.write(json.dumps({"messages": messages}) + "\n")
            self.process.stdin.flush()
            try:
                return self._read(timeout=max(600, 15 * len(messages)))["predictions"]
            except (RuntimeError, TimeoutError, ValueError, KeyError):
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.process:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                for stream in (self.process.stdin, self.process.stdout):
                    if stream:
                        stream.close()
                self.process = None
            if self.log:
                self.log.close()
                self.log = None
            self.model_key = None
