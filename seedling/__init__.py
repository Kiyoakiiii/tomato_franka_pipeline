"""Tomato seedling transplanting pipeline (FR3 + ZED 2i + parallel gripper).

Frame naming used throughout: ``T_AB`` maps coordinates expressed in frame B
into frame A, i.e. ``p_A = T_AB @ p_B``.

    B  robot base (franky / libfranka base frame), metres
    E  franky end-effector frame (as configured in Franka Desk)
    G  gripper contact centre: z = approach direction, y = closing direction
    C  ZED left rectified camera, IMAGE convention: x right, y down, z forward
"""
