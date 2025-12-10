"""
Graphical joint monitor for the Unitree G1 robot.

Displays per-joint angles, operating ranges, and dual temperature probes with
color-coding and safety blinking once the core (T0) exceeds 55 °C.
"""

from __future__ import annotations

import argparse
import sys

try:
    from PyQt5 import QtWidgets  # type: ignore
except ImportError as exc:  # pragma: no cover - GUI dependency check
    print(
        "monitor_g1.py requires PyQt5. Install it with `pip install PyQt5` and try again.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from const import DEFAULT_INTERFACE
from utilities import ensure_channel_factory
from gui_widgets import JointStateMonitorWindow, StateReceiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise Unitree G1 joint telemetry with live safety indicators."
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=DEFAULT_INTERFACE,
        help="Network interface name for ChannelFactoryInitialize (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_channel_factory(args.interface)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Unitree G1 Joint Monitor")

    receiver = StateReceiver()
    window = JointStateMonitorWindow(receiver)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
