import os
import sys

sys.dont_write_bytecode = True

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def cfg():
    from seedling.config import load_config

    c = load_config()
    c["grasp"]["table_z_m"] = 0.05  # synthetic tray top
    return c
