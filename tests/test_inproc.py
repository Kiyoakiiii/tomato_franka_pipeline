"""The run's steps in one process (2026-11-10): one robot connection for all of them, and a step's exit code as its
own process would have given it."""
import sys
import types

import numpy as np


def _fake_franky(made):
    fr = types.ModuleType("franky")

    class Robot:
        def __init__(self, host, realtime_config=None):
            made.append(host)
            self.current_joint_positions = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
            self.relative_dynamics_factor = None

        def set_collision_behavior(self, *a):
            pass

    fr.Robot = Robot
    fr.RealtimeConfig = types.SimpleNamespace(Enforce=1, Ignore=0)
    fr.RelativeDynamicsFactor = lambda *a: a
    return fr


def test_the_steps_of_a_run_share_one_robot_connection(cfg, monkeypatch):
    from seedling import robot as R

    made = []
    monkeypatch.setitem(sys.modules, "franky", _fake_franky(made))
    monkeypatch.setattr(R.FrankyRobot, "_check_ee_frame", lambda self: None)
    monkeypatch.delenv(R.STATE_ENV, raising=False)
    monkeypatch.delenv(R.SIM_ENV, raising=False)
    monkeypatch.setattr(R, "_SHARED", {})
    monkeypatch.setenv(R.SHARED_ENV, "1")
    a = R.FrankyRobot(cfg, step_mode=False)
    b = R.FrankyRobot(cfg, step_mode=True)
    assert a is b and len(made) == 1 and b.step_mode is True
    np.testing.assert_allclose(b.q(), [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    # without the variable (a script on its own) every robot is its own connection, as before
    monkeypatch.delenv(R.SHARED_ENV)
    c = R.FrankyRobot(cfg, step_mode=False)
    assert c is not a and len(made) == 2


def test_a_step_run_in_process_gives_the_exit_code_its_process_would(tmp_path, monkeypatch):
    from seedling import inproc

    monkeypatch.setattr(inproc, "PROJECT_DIR", tmp_path)
    (tmp_path / "s_ok.py").write_text("import sys\ndef main():\n    print('hello', sys.argv[1:])\n    return 0\n")
    (tmp_path / "s_three.py").write_text("def main():\n    return 3\n")
    (tmp_path / "s_exit.py").write_text("import sys\ndef main():\n    sys.exit(2)\n")
    (tmp_path / "s_raise.py").write_text("def main():\n    raise RuntimeError('boom')\n")
    rc, out = inproc.run_script("s_ok.py", ["--cell", "C5"], quiet=True)
    assert rc == 0 and "hello ['--cell', 'C5']" in out
    assert inproc.run_script("s_three.py", []) == 3
    assert inproc.run_script("s_exit.py", []) == 2
    rc, out = inproc.run_script("s_raise.py", [], quiet=True)
    assert rc == 1 and "RuntimeError: boom" in out
