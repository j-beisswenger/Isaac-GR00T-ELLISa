#!/usr/bin/env python3

"""Full-screen live preview of the front RGB camera."""

import argparse
import signal
import sys
from PyQt5 import QtWidgets
from const import DEFAULT_CAMERA_FPS, DEFAULT_INTERFACE
from utilities import create_video_client
from gui_widgets import CameraWidget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fullscreen preview of the G1 front RGB camera."
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=DEFAULT_INTERFACE,
        help="Network interface for ChannelFactoryInitialize (default: %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help="Target refresh rate for the preview widget (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = create_video_client(args.interface)

    app = QtWidgets.QApplication(sys.argv)
    widget = CameraWidget(client, fps=args.fps)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    widget.resize(1280, 720)
    widget.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
