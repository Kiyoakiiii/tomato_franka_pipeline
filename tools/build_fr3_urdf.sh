#!/bin/bash
# Franka's own FR3 description for the collision model (seedling/world.py): data/third_party/fr3_build/fr3.urdf
# (data/ is not in git). Needs network once; xacro comes from the project venv (pip install xacro).
set -e
cd "$(dirname "$0")/.."
mkdir -p data/third_party && cd data/third_party
[ -d franka_description ] || git clone --depth 1 https://github.com/frankaemika/franka_description.git
rm -rf fr3_build && cp -r franka_description fr3_build && rm -rf fr3_build/.git
D=$PWD/fr3_build
grep -rl 'find franka_description' fr3_build --include=*.xacro | xargs sed -i "s|\$(find franka_description)|$D|g"
cd fr3_build && ../../../.venvs/pipeline/bin/xacro robots/fr3/fr3.urdf.xacro hand:=false > fr3.urdf
echo "written: data/third_party/fr3_build/fr3.urdf"
