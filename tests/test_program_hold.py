import numpy as np


def _prog(cfg, off):
    from seedling.program import Program

    cfg["grasp"]["side_offset_mm"] = list(off)
    cells = {f"{r}{c}": [0.70 + 0.0448 * i, 0.25 - 0.0448 * (c - 1), 0.18] for i, r in enumerate("ABCDEF") for c in range(1, 13)}
    grid = {"cells": cells, "z_top": 0.18}
    holes = [[0.58 + 0.057 * np.cos(t), 0.0 + 0.057 * np.sin(t)] for t in np.radians(np.arange(0, 360, 60))]
    disc = {"holes": holes, "z_top": 0.06}
    return Program(cfg, np.eye(4), grid, disc)


def test_the_clamped_stem_goes_over_the_hole_whatever_the_grasp_offset(cfg):
    """the pins clamp the stem where they meet (the closing point); whatever offset the closing point was given from the
    cell centre, the placement brings that clamped point over the hole (2026-11-06: modelling the stem 10 mm past the
    closing point put every plant 10 mm off its hole)"""
    for off in ([0.0, 0.0], [-10.0, 0.0], [6.0, -4.0]):
        p = _prog(cfg, off)
        for cell in ("C9", "A1", "F12"):
            P = p.cell_poses(cell, 0.20)
            Tg = P["grasp"]
            clamp_in_hand = Tg[:3, :3].T @ (P["close"] - Tg[:3, 3])
            for hole in range(1, 7):
                for dyaw in (0.0, 30.0, -50.0):
                    Ti = p.hole_poses(hole, dyaw, 0.06)["insert"]
                    clamp = Ti[:3, 3] + Ti[:3, :3] @ clamp_in_hand
                    assert np.allclose(clamp[:2], p.holes[hole - 1][:2], atol=1e-9), (off, cell, hole, dyaw)


def test_the_plug_length_comes_from_the_lowest_soil_reading():
    """close-up and overview disagree: the grasp height from the higher, the plug's length from the lower (a plug taken
    shorter than it is reaches the disc before the hand is down)"""
    import importlib

    m = importlib.import_module("plan")
    grid = {"soil_z_overview": {"C8": 0.197}}
    obs = {"measure": {"ok": True, "soil_z": 0.215, "shift_mm": [1.0, 1.0]}}
    si = m.soil_info(grid, "C8", obs, 0.182)
    assert si["soil"] == 0.215 and si["soil_low"] == 0.197



def test_a_pitched_hand_brings_the_clamped_stem_over_the_hole_and_keeps_it_upright(cfg):
    """a grasp made with the hand pitched down (deep in the tray) keeps that pitch to the hole: the clamped point over
    the hole centre, and the hand's attitude at the hole the same as at the grasp up to the turn about the vertical
    (so the stem, vertical at the grasp, is vertical when it is let go)"""
    from seedling.program import GRASP_PITCHES

    p = _prog(cfg, [0.0, 0.0])
    for pitch in GRASP_PITCHES:
        P = p.cell_poses("D8", 0.20, pitch_deg=pitch)
        Tg = P["grasp"]
        assert np.isclose(np.degrees(np.arcsin(-Tg[2, 2])), pitch)  # the approach axis pitched down
        clamp_in_hand = Tg[:3, :3].T @ (P["close"] - Tg[:3, 3])
        up_in_hand = Tg[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        for hole in (1, 4):
            Ti = p.hole_poses(hole, 20.0, 0.06, pitch_deg=pitch)["insert"]
            clamp = Ti[:3, 3] + Ti[:3, :3] @ clamp_in_hand
            assert np.allclose(clamp[:2], p.holes[hole - 1][:2], atol=1e-9)
            assert np.allclose(Ti[:3, :3] @ up_in_hand, [0.0, 0.0, 1.0], atol=1e-9)
