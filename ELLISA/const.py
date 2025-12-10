"""
Configuration constants and shared settings for the G1 control panel scripts.

This module centralizes all configuration values used across the control-panel
scripts (keyboard teleop, monitor, camera viewer).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

# ============================================================================
# Network Configuration
# ============================================================================

DEFAULT_INTERFACE = "enx00e04c3e5a60"
DEFAULT_CAMERA_FPS = 15


# ============================================================================
# Locomotion Configuration
# ============================================================================

DEFAULT_STAND_HEIGHT = 0.25  # metres – gentle bring-up for ground start
READY_FSM_IDS = (200, 204, 205, 500)
READY_MODES = (0, 1, 3, 4)


# ============================================================================
# Keyboard Teleop Configuration
# ============================================================================

# Conservative defaults so the robot only takes small steps
FORWARD_SPEED = 1.50  # m/s
LATERAL_SPEED = 0.90  # m/s
YAW_RATE = 2.00  # rad/s
SEND_PERIOD = 0.10  # seconds between Move() refreshes
STATE_POLL_PERIOD = 0.5  # seconds between state queries
ROTATION_PULSE_OMEGA = math.pi / 4  # rad/s for ~45deg turn over 1s pulse
MODE_HOLD_DURATION = 2.0  # seconds to hold for mode change
MODE_HOLD_INTERVAL = 0.15  # seconds between repeated mode commands
TASK_HOLD_DURATION = 0.6  # seconds to hold for task trigger
TASK_HOLD_INTERVAL = 0.15  # seconds between repeated task commands
DOUBLE_PRESS_WINDOW = 1.5  # seconds window for double-press detection


# ============================================================================
# Joint Configuration (for monitoring GUI)
# ============================================================================

# Joint limits in radians (min, max)
JOINT_LIMITS: Dict[int, Tuple[float, float]] = {
    0: (-2.5307, 2.8798),
    1: (-0.5236, 2.9671),
    2: (-2.7576, 2.7576),
    3: (-0.087267, 2.8798),
    4: (-0.87267, 0.5236),
    5: (-0.2618, 0.2618),
    6: (-2.5307, 2.8798),
    7: (-2.9671, 0.5236),
    8: (-2.7576, 2.7576),
    9: (-0.087267, 2.8798),
    10: (-0.87267, 0.5236),
    11: (-0.2618, 0.2618),
    12: (-2.618, 2.618),
    13: (-0.52, 0.52),
    14: (-0.52, 0.52),
    15: (-3.0892, 2.6704),
    16: (-1.5882, 2.2515),
    17: (-2.618, 2.618),
    18: (-1.0472, 2.0944),
    19: (-1.972222054, 1.972222054),
    20: (-1.614429558, 1.614429558),
    21: (-1.614429558, 1.614429558),
    22: (-3.0892, 2.6704),
    23: (-2.2515, 1.5882),
    24: (-2.618, 2.618),
    25: (-1.0472, 2.0944),
    26: (-1.972222054, 1.972222054),
    27: (-1.614429558, 1.614429558),
    28: (-1.614429558, 1.614429558),
}

# Gauge visualization limits
GAUGE_GLOBAL_MIN = min(limits[0] for limits in JOINT_LIMITS.values())
GAUGE_GLOBAL_MAX = max(limits[1] for limits in JOINT_LIMITS.values())
GAUGE_GLOBAL_SPAN = max(GAUGE_GLOBAL_MAX - GAUGE_GLOBAL_MIN, 1e-6)


# ============================================================================
# Joint Groupings (for monitoring UI)
# ============================================================================

JOINT_GROUPS: Dict[str, Sequence[Tuple[str, int]]] = {
    "🦵 Left Leg": (
        ("LeftHipPitch", 0),
        ("LeftHipRoll", 1),
        ("LeftHipYaw", 2),
        ("LeftKnee", 3),
        ("LeftAnklePitch", 4),
        ("LeftAnkleRoll", 5),
    ),
    "🦵 Right Leg": (
        ("RightHipPitch", 6),
        ("RightHipRoll", 7),
        ("RightHipYaw", 8),
        ("RightKnee", 9),
        ("RightAnklePitch", 10),
        ("RightAnkleRoll", 11),
    ),
    "🧍 Waist": (
        ("WaistYaw", 12),
        ("WaistRoll", 13),
        ("WaistPitch", 14),
    ),
    "🫲 Left Arm": (
        ("LeftShoulderPitch", 15),
        ("LeftShoulderRoll", 16),
        ("LeftShoulderYaw", 17),
        ("LeftElbow", 18),
        ("LeftWristRoll", 19),
        ("LeftWristPitch", 20),
        ("LeftWristYaw", 21),
    ),
    "🫱 Right Arm": (
        ("RightShoulderPitch", 22),
        ("RightShoulderRoll", 23),
        ("RightShoulderYaw", 24),
        ("RightElbow", 25),
        ("RightWristRoll", 26),
        ("RightWristPitch", 27),
        ("RightWristYaw", 28),
    ),
}


# ============================================================================
# Temperature Configuration
# ============================================================================

TEMP_BLUE_THRESHOLD = 37.0
TEMP_RED_THRESHOLD = 45.0
TEMP_BLINK_THRESHOLD = 55.0


# ============================================================================
# UI Color Configuration
# ============================================================================

COLOR_GAUGE_POSITIVE = "#9467bd"
COLOR_GAUGE_NEGATIVE = "#17becf"
COLOR_GAUGE_ZERO = "#7f7f7f"
COLOR_GAUGE_LIMIT = "#ff7f0e"
COLOR_GAUGE_EXCEEDED = "#d62728"


# ============================================================================
# Font Configuration (for monitoring UI)
# ============================================================================

UI_FONT_CHOICES = ["Segoe UI", "Noto Sans", "Roboto", "Helvetica Neue", "Arial"]
EMOJI_FONT_CHOICES = [
    "Segoe UI Emoji",
    "Apple Color Emoji",
    "Noto Color Emoji",
    "Noto Emoji",
    "Noto Sans",
]
MONO_FONT_CHOICES = [
    "Roboto Mono",
    "Noto Sans Mono",
    "DejaVu Sans Mono",
    "Menlo",
    "Consolas",
    "Courier New",
]
