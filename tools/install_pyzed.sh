#!/usr/bin/env bash
# Install the ZED Python API (pyzed) into the project env .venvs/pipeline.
# Requires read access to /usr/local/zed (account in group 'zed'). Touches nothing outside the project.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venvs/pipeline/bin/python"
export PIP_CACHE_DIR="$PWD/.cache/pip" TMPDIR="$PWD/.tmp"
if [ ! -r /usr/local/zed/get_python_api.py ]; then
  echo "cannot read /usr/local/zed/get_python_api.py — ask the lab admin to run: sudo usermod -aG zed $USER (then log in again)" >&2
  exit 1
fi
# SDK 5.3 ships a ready wheel for this Python; otherwise let the SDK script download the matching one
WHL=$(ls /usr/local/zed/pyzed-*-cp312-cp312-linux_x86_64.whl 2>/dev/null | head -1 || true)
if [ -n "$WHL" ]; then
  "$PY" -m pip install "$WHL"
else
  mkdir -p .tmp/pyzed && cd .tmp/pyzed
  "$PY" /usr/local/zed/get_python_api.py
fi
"$PY" -c "import pyzed.sl as sl; print('pyzed OK, SDK', sl.Camera.get_sdk_version())"
