#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-base}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CPU_TORCH_INDEX_URL="${CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
CUDA_TORCH_SPEC="${CUDA_TORCH_SPEC:-torch==${TORCH_VERSION}+cu124}"
CUDA_TORCHVISION_SPEC="${CUDA_TORCHVISION_SPEC:-torchvision==${TORCHVISION_VERSION}+cu124}"
CUDA_TORCHAUDIO_SPEC="${CUDA_TORCHAUDIO_SPEC:-torchaudio==${TORCHAUDIO_VERSION}+cu124}"
CPU_TORCH_SPEC="${CPU_TORCH_SPEC:-torch==${TORCH_VERSION}+cpu}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv or add it to PATH." >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ -d "$VENV_DIR/bin" ]; then
  echo "Using existing virtual environment at: $VENV_DIR"
else
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
uv pip install --index-url "$PYPI_INDEX_URL" --upgrade pip setuptools wheel

case "$PROFILE" in
  base)
    uv pip install --index-url "$PYPI_INDEX_URL" -e .
    ;;
  dev)
    uv pip install --index-url "$PYPI_INDEX_URL" -e '.[dev]'
    ;;
  analysis)
    uv pip install --index-url "$PYPI_INDEX_URL" -e '.[dev,analysis]'
    ;;
  cpu-ml)
    uv pip install "$CPU_TORCH_SPEC" --index-url "$CPU_TORCH_INDEX_URL" --extra-index-url "$PYPI_INDEX_URL" --index-strategy unsafe-best-match
    uv pip install --index-url "$PYPI_INDEX_URL" --extra-index-url "$CPU_TORCH_INDEX_URL" --index-strategy unsafe-best-match -e '.[dev,analysis,ml]' "$CPU_TORCH_SPEC"
    ;;
  ml)
    uv pip install "$CUDA_TORCH_SPEC" "$CUDA_TORCHVISION_SPEC" "$CUDA_TORCHAUDIO_SPEC" --index-url "$CUDA_INDEX_URL" --extra-index-url "$PYPI_INDEX_URL" --index-strategy unsafe-best-match
    uv pip install --index-url "$PYPI_INDEX_URL" --extra-index-url "$CUDA_INDEX_URL" --index-strategy unsafe-best-match -e '.[dev,analysis,ml]' "$CUDA_TORCH_SPEC"
    ;;
  serve)
    uv pip install "$CUDA_TORCH_SPEC" "$CUDA_TORCHVISION_SPEC" "$CUDA_TORCHAUDIO_SPEC" --index-url "$CUDA_INDEX_URL" --extra-index-url "$PYPI_INDEX_URL" --index-strategy unsafe-best-match
    uv pip install --index-url "$PYPI_INDEX_URL" --extra-index-url "$CUDA_INDEX_URL" --index-strategy unsafe-best-match -e '.[dev,analysis,ml,serve]' "$CUDA_TORCH_SPEC"
    ;;
  all)
    uv pip install "$CUDA_TORCH_SPEC" "$CUDA_TORCHVISION_SPEC" "$CUDA_TORCHAUDIO_SPEC" --index-url "$CUDA_INDEX_URL" --extra-index-url "$PYPI_INDEX_URL" --index-strategy unsafe-best-match
    uv pip install --index-url "$PYPI_INDEX_URL" --extra-index-url "$CUDA_INDEX_URL" --index-strategy unsafe-best-match -e '.[all]' "$CUDA_TORCH_SPEC"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Usage: scripts/setup_env.sh [base|dev|analysis|cpu-ml|ml|serve|all]" >&2
    exit 1
    ;;
esac

python - <<'PY'
import importlib.util
import sys

print("python", sys.version.split()[0], sys.executable)
for name in ["torch", "transformers", "xgrammar", "jsonschema", "pandas", "pytest"]:
    print(f"{name}: {'ok' if importlib.util.find_spec(name) else 'missing'}")
PY
