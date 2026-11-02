#!/usr/bin/env bash
# Builds the project-local runtime env. Everything stays under the project dir.
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.cache/pip" XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/.tmp"
WHL=franky_control-2.0.0+libfranka.0.21.2-cp312-cp312-manylinux_2_34_x86_64.whl
if [ ! -f ".tmp/wheels/$WHL" ]; then
  curl -fL -o ".tmp/wheels/$WHL" "https://github.com/TimSchneider42/franky/releases/download/v2.0.0/${WHL//+/%2B}"
fi
[ -x .venvs/pipeline/bin/python ] || python3 -m venv .venvs/pipeline
P=.venvs/pipeline/bin/python
$P -m pip install --upgrade pip==24.0
$P -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 ultralytics==8.4.46 numpy==2.2.6 \
  opencv-python==5.0.0.93 scipy==1.18.1 scikit-image pyserial pytest matplotlib
$P -m pip install ".tmp/wheels/$WHL"
$P -m pip install open3d || echo "open3d install failed (optional)"
# TensorRT 10 for ZED NEURAL depth (not installed system-wide); unversioned names are what the SDK dlopens
$P -m pip install "tensorrt-cu12-libs==10.16.1.11"
TRT=$($P -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))")
mkdir -p .venvs/pipeline/trt_links
for n in libnvinfer libnvinfer_plugin libnvonnxparser; do
  ln -sf "$TRT/$n.so.10" ".venvs/pipeline/trt_links/$n.so"; ln -sf "$TRT/$n.so.10" ".venvs/pipeline/trt_links/$n.so.10"
done
# the TensorRT builder loads the per-GPU resource library by name when a model is first optimised
for f in "$TRT"/libnvinfer_builder_resource_*.so.10.*; do ln -sf "$f" ".venvs/pipeline/trt_links/$(basename "$f")"; done
# pyzed comes from the ZED SDK (needs group zed): tools/install_pyzed.sh
# one-time NEURAL optimisation for this GPU (writes into /usr/local/zed/resources):
#   LD_LIBRARY_PATH=$PWD/.venvs/pipeline/trt_links /usr/local/zed/tools/ZED_Diagnostic -nrlo
$P -m pip freeze > requirements-pipeline.lock.txt
echo DONE
