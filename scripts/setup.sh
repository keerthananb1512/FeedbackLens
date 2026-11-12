#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'Install uv, then rerun this script.' >&2; exit 1; }
uv venv --python 3.11 --allow-existing .venv
uv pip install --python .venv/bin/python torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
# Explicit lock avoids optional xformers/torchao upgrades replacing the tested torch build.
uv pip install --python .venv/bin/python --no-deps -r requirements-runtime.lock
uv pip install --python .venv/bin/python --no-deps -e .
# Build/use a wheel matched to the torch/CUDA ABI. This is the official release wheel
# used by this run, not a source build that might download another torch in isolation.
uv pip install --python .venv/bin/python --no-deps 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
# d1 needs a newer Transformers than Unsloth. Reuse the same torch and core dependencies.
uv venv --python .venv/bin/python --system-site-packages --allow-existing .venv-d1
.venv-d1/bin/python - <<'PY'
from pathlib import Path
import sysconfig
root = Path.cwd()
Path(sysconfig.get_path('purelib'), 'feedbacklens_runtime.pth').write_text(
    str(root / '.venv/lib/python3.11/site-packages') + '\n'
)
PY
uv pip install --python .venv-d1/bin/python --no-deps -r requirements-d1.txt
