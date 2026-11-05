import json
import sys

import numpy as np
import pytest

from seedling.config import load_config


def test_sim_env_never_connects(tmp_path, monkeypatch):
    q = [0.1, -0.8, -0.2, -2.0, -0.1, 1.6, 0.7]
    f = tmp_path / "q.json"
    f.write_text(json.dumps({"q": q}))
    monkeypatch.setenv("SEEDLING_SIM_Q", str(f))
    from seedling.robot import FrankyRobot, PlanningOnlyRobot

    r = FrankyRobot(load_config(), step_mode=False)
    assert isinstance(r, PlanningOnlyRobot)
    assert np.allclose(r.q(), q)
    with pytest.raises(SystemExit):
        r.move_joints(np.array(q))
    assert "franky" not in sys.modules
