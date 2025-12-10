"""
GUI widgets and components for the G1 control panel applications.

This module provides PyQt5-based widgets for visualising robot state, camera
feeds, and joint telemetry for the Unitree G1 robot.  The implementation mirrors
the original monitor_g1.py look-and-feel so refactored entry points share the
same UI.
"""

from __future__ import annotations

import math
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from const import (
    COLOR_GAUGE_EXCEEDED,
    COLOR_GAUGE_LIMIT,
    COLOR_GAUGE_NEGATIVE,
    COLOR_GAUGE_POSITIVE,
    COLOR_GAUGE_ZERO,
    DEFAULT_CAMERA_FPS,
    EMOJI_FONT_CHOICES,
    GAUGE_GLOBAL_MAX,
    GAUGE_GLOBAL_MIN,
    GAUGE_GLOBAL_SPAN,
    JOINT_GROUPS,
    JOINT_LIMITS,
    MONO_FONT_CHOICES,
    TEMP_BLINK_THRESHOLD,
    TEMP_BLUE_THRESHOLD,
    TEMP_RED_THRESHOLD,
    UI_FONT_CHOICES,
)

from utilities import create_video_client

try:  # Access to the VideoClient class is optional; only needed for UI toggles.
    from utilities import VideoClient  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - VideoClient is hardware dependent.
    VideoClient = None  # type: ignore[assignment]

try:  # Unitree SDK telemetry types are required for live state visualisation.
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, LowState_
except ImportError:  # pragma: no cover - allows importing without SDK.
    ChannelSubscriber = None  # type: ignore[assignment]
    BmsState_ = None  # type: ignore[assignment]
    LowState_ = None  # type: ignore[assignment]

try:  # Optional GUI dependency; functions raise helpful errors when missing.
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
except ImportError:  # pragma: no cover - GUI tooling may be absent on headless systems.
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]


# ============================================================================
# Optional OpenCV cache (avoids import conflicts with Qt plugins)
# ============================================================================

_CV2_MODULE: Optional[Any] = None
_CV2_TRIED_IMPORT = False


# ============================================================================
# Helper functions shared across widgets
# ============================================================================


def _require_pyqt() -> None:
    """Ensure PyQt5 is available before constructing GUI components."""
    if QtWidgets is None:  # pragma: no cover - depends on runtime environment.
        raise RuntimeError(
            "PyQt5 is required for GUI components but is not installed."
        )


def camera_frame_interval_ms(fps: float = DEFAULT_CAMERA_FPS) -> int:
    """Convert frames-per-second into a QTimer-compatible millisecond interval."""
    return max(1, int(1000 / max(fps, 1e-6)))


def _ensure_cv2() -> Optional[Any]:
    """Import OpenCV on demand so it does not interfere with Qt plugin loading."""

    global _CV2_MODULE, _CV2_TRIED_IMPORT
    if _CV2_TRIED_IMPORT:
        return _CV2_MODULE

    try:
        import cv2 as _cv2  # type: ignore
    except ImportError:  # pragma: no cover - recording fallback.
        _CV2_MODULE = None
    else:
        _CV2_MODULE = _cv2
    finally:
        _CV2_TRIED_IMPORT = True
    return _CV2_MODULE


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value into the provided inclusive bounds."""
    return max(lower, min(value, upper))


def _safe_float(value: object) -> float:
    """Convert to float, returning NaN when the input is not representable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def temperature_color(temp: float) -> "QtGui.QColor":
    """Return a Qt color instance matching the legacy temperature thresholds."""
    if QtGui is None:
        raise RuntimeError("temperature_color requires PyQt5.")
    if math.isnan(temp):
        return QtGui.QColor("#888888")
    if temp <= TEMP_BLUE_THRESHOLD:
        return QtGui.QColor("#1f77b4")
    if temp >= TEMP_RED_THRESHOLD:
        return QtGui.QColor("#d62728")
    return QtGui.QColor("#101010")


# ============================================================================
# DDS telemetry helpers
# ============================================================================


class StateReceiver:
    """Background subscriber feeding the GUI with low-state frames."""

    def __init__(self, topic: str = "rt/lowstate") -> None:
        if ChannelSubscriber is None or LowState_ is None:
            raise RuntimeError("Unitree SDK low-state messages are unavailable.")
        self._topic = topic
        self._latest: Optional[LowState_] = None
        self._lock = threading.Lock()
        self._subscriber = ChannelSubscriber(topic, LowState_)
        self._subscriber.Init(self._handle_state, 10)

    def _handle_state(self, msg: LowState_) -> None:
        with self._lock:
            self._latest = msg

    def consume_latest(self) -> Optional[LowState_]:
        with self._lock:
            state = self._latest
            self._latest = None
        return state

    def get_state(self) -> Optional[LowState_]:
        """Return the latest state without consuming it."""
        with self._lock:
            return self._latest


class BatteryReceiver:
    """Optional subscriber for battery power state frames."""

    def __init__(self, topic: str = "rt/bmsstate") -> None:
        if ChannelSubscriber is None or BmsState_ is None:
            raise RuntimeError("Unitree SDK BMS telemetry is unavailable.")
        self._topic = topic
        self._latest: Optional[BmsState_] = None
        self._lock = threading.Lock()
        self._active = False
        self._error: Optional[str] = None
        try:
            self._subscriber = ChannelSubscriber(topic, BmsState_)
            self._subscriber.Init(self._handle_state, 10)
            self._active = True
        except Exception as exc:  # pragma: no cover - depends on HW topics
            self._subscriber = None
            self._error = str(exc)

    def _handle_state(self, msg: BmsState_) -> None:
        with self._lock:
            self._latest = msg

    def consume_latest(self) -> Optional[BmsState_]:
        if not self._active:
            return None
        with self._lock:
            msg = self._latest
            self._latest = None
        return msg

    @property
    def active(self) -> bool:
        return self._active

    @property
    def error(self) -> Optional[str]:
        return self._error


# ============================================================================
# Qt-based widgets
# ============================================================================


if QtWidgets is not None:

    def make_font(
        size: int,
        *,
        bold: bool = False,
        emoji: bool = False,
    ) -> QtGui.QFont:
        """Create a UI font with the same fallback list as the legacy monitor."""
        families = EMOJI_FONT_CHOICES if emoji else UI_FONT_CHOICES
        for family in families:
            font = QtGui.QFont(family, pointSize=size)
            if QtGui.QFontInfo(font).family():
                font.setBold(bold)
                return font
        font = QtGui.QFont()
        font.setPointSize(size)
        font.setBold(bold)
        return font

    def make_mono_font(size: int, *, bold: bool = False) -> QtGui.QFont:
        """Return a monospace font from the preferred list."""
        for family in MONO_FONT_CHOICES:
            font = QtGui.QFont(family, pointSize=size)
            if QtGui.QFontInfo(font).family():
                font.setBold(bold)
                return font
        font = QtGui.QFont("Monospace", pointSize=size)
        font.setBold(bold)
        font.setStyleHint(QtGui.QFont.TypeWriter)
        return font

    class CameraWidget(QtWidgets.QWidget):
        """Full-screen camera preview widget with optional recording support."""

        def __init__(self, client: Any, fps: float = DEFAULT_CAMERA_FPS):
            _require_pyqt()
            super().__init__()
            self._client = client
            self._target_fps = max(float(fps), 1e-3)
            self._frame_interval = camera_frame_interval_ms(self._target_fps)
            self._base_title = "ELLISa0 G1 Front RGB Camera"

            self._recording = False
            self._video_writer: Optional[Any] = None
            self._record_path: Optional[Path] = None
            self._last_frame_size: Optional[Tuple[int, int]] = None
            self._cv2: Optional[Any] = None

            self._label = QtWidgets.QLabel("Connecting to camera…", self)
            self._label.setAlignment(QtCore.Qt.AlignCenter)
            self._default_stylesheet = self._label.styleSheet()

            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(self._label)

            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._update_frame)
            self._timer.start(self._frame_interval)

            self._escape_shortcut = QtWidgets.QShortcut(
                QtGui.QKeySequence("Escape"), self
            )
            self._escape_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            self._escape_shortcut.activated.connect(self._close_requested)

            self._record_shortcut = QtWidgets.QShortcut(
                QtGui.QKeySequence("R"), self
            )
            self._record_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            self._record_shortcut.activated.connect(self._toggle_recording)

            self._update_window_title()

        def _close_requested(self) -> None:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.quit()

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
            self._timer.stop()
            self._stop_recording(finalising=True)
            super().closeEvent(event)

        def _update_frame(self) -> None:
            try:
                code, data = self._client.GetImageSample()
            except Exception as exc:  # pragma: no cover - depends on hardware.
                self._label.setText(f"Camera error: {exc}")
                return

            if code != 0 or not data:
                self._label.setText(f"Camera unavailable (code={code})")
                return

            frame_bytes = bytes(data)
            image = QtGui.QImage.fromData(frame_bytes)
            if image.isNull():
                self._label.setText("Failed to decode camera frame")
                return

            pixmap = QtGui.QPixmap.fromImage(image)
            if pixmap.isNull():
                self._label.setText("Failed to convert camera frame")
                return

            self._handle_recording_frame(frame_bytes)

            target_size = self._label.size()
            if target_size.width() == 0 or target_size.height() == 0:
                display_pixmap = pixmap
            else:
                display_pixmap = pixmap.scaled(
                    target_size,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                if display_pixmap.isNull():
                    display_pixmap = pixmap

            self._label.setPixmap(display_pixmap)

        def _toggle_recording(self) -> None:
            if self._recording:
                self._stop_recording()
                return

            cv2_mod = _ensure_cv2()
            if cv2_mod is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Recording unavailable",
                    "Video recording requires OpenCV (cv2). Install it to enable recording.",
                )
                return

            self._cv2 = cv2_mod
            self._start_recording()

        def _start_recording(self) -> None:
            if self._recording:
                return

            output_dir = Path.cwd() / "out_videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._record_path = output_dir / f"camera_{timestamp}.mp4"
            self._recording = True
            self._video_writer = None
            self._last_frame_size = None
            self._label.setStyleSheet("border: 4px solid red;")
            self._update_window_title()
            print(f"[CameraWidget] Recording started: {self._record_path}")

        def _stop_recording(self, *, finalising: bool = False) -> None:
            if not self._recording and self._video_writer is None:
                return

            self._recording = False
            writer = self._video_writer
            self._video_writer = None
            frame_size = self._last_frame_size
            self._last_frame_size = None
            path = self._record_path
            self._record_path = None

            if writer is not None:
                writer.release()

            self._label.setStyleSheet(self._default_stylesheet)
            self._update_window_title()

            if path and not finalising:
                if frame_size is None:
                    print("[CameraWidget] Recording stopped without frames.")
                else:
                    print(f"[CameraWidget] Recording saved: {path}")

        def _handle_recording_frame(self, frame_bytes: bytes) -> None:
            if not self._recording or self._cv2 is None:
                return

            try:
                cv2 = self._cv2
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            except Exception as exc:  # pragma: no cover - depends on cv2 install.
                print(
                    f"[CameraWidget] Failed to decode frame for recording: {exc}",
                    file=sys.stderr,
                )
                return

            if frame is None:
                print(
                    "[CameraWidget] Camera frame decoding returned None.",
                    file=sys.stderr,
                )
                return

            height, width = frame.shape[:2]
            if self._video_writer is None:
                if not self._initialise_writer(width, height):
                    self._stop_recording()
                    return
            elif self._last_frame_size and (width, height) != self._last_frame_size:
                try:
                    frame = cv2.resize(
                        frame,
                        (
                            self._last_frame_size[0],
                            self._last_frame_size[1],
                        ),
                    )
                except Exception as exc:  # pragma: no cover - resize failure rare.
                    print(
                        f"[CameraWidget] Failed to resize frame for recording: {exc}",
                        file=sys.stderr,
                    )
                    return

            if self._video_writer is not None:
                self._video_writer.write(frame)

        def _initialise_writer(self, width: int, height: int) -> bool:
            if self._record_path is None or self._cv2 is None:
                return False

            try:
                cv2 = self._cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self._record_path),
                    fourcc,
                    float(self._target_fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("VideoWriter failed to open.")
            except Exception as exc:  # pragma: no cover - depends on cv2 support.
                print(
                    f"[CameraWidget] Failed to start recording: {exc}",
                    file=sys.stderr,
                )
                return False

            self._video_writer = writer
            self._last_frame_size = (width, height)
            return True

        def _update_window_title(self) -> None:
            title = self._base_title
            if self._recording:
                title += " [REC]"
            self.setWindowTitle(title)

    class AngleGauge(QtWidgets.QWidget):
        """Custom gauge that visualises joint angle across its limits."""

        def __init__(self) -> None:
            super().__init__()
            self._angle = float("nan")
            self._limits: Optional[Tuple[float, float]] = None
            self._inactive = False
            self.setFixedHeight(26)
            self.setMinimumWidth(170)

        def update_value(
            self, angle_rad: float, limits: Optional[Tuple[float, float]]
        ) -> None:
            self._angle = angle_rad
            self._limits = limits
            self.update()

        def set_inactive(self, inactive: bool) -> None:
            if self._inactive != inactive:
                self._inactive = inactive
                self.update()

        @property
        def out_of_bounds(self) -> bool:
            if self._limits is None or math.isnan(self._angle):
                return False
            min_lim, max_lim = self._limits
            return self._angle < min_lim or self._angle > max_lim

        def _map_global(self, value: float) -> float:
            return _clamp((value - GAUGE_GLOBAL_MIN) / GAUGE_GLOBAL_SPAN, 0.0, 1.0)

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            rect = self.rect().adjusted(4, 6, -4, -6)
            base_color = QtGui.QColor("#f0f0f0" if not self._inactive else "#f7f7f7")
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(base_color)
            painter.drawRoundedRect(rect, 8, 8)

            if self._limits is None:
                return

            min_lim, max_lim = self._limits
            if max_lim < min_lim:
                min_lim, max_lim = max_lim, min_lim

            left_ratio = _clamp(self._map_global(min_lim), 0.0, 1.0)
            right_ratio = _clamp(self._map_global(max_lim), 0.0, 1.0)
            left_x = rect.left() + left_ratio * rect.width()
            right_x = rect.left() + right_ratio * rect.width()
            if right_x < left_x:
                left_x, right_x = right_x, left_x
            range_width = max(2.0, right_x - left_x)

            painter.setOpacity(0.5 if self._inactive else 0.8)
            painter.setBrush(
                QtGui.QColor("#d9d9d9" if not self._inactive else "#ededed")
            )
            painter.drawRoundedRect(
                QtCore.QRectF(left_x, rect.top(), range_width, rect.height()), 6, 6
            )

            if min_lim < 0.0 < max_lim:
                zero_ratio = _clamp(self._map_global(0.0), 0.0, 1.0)
                zero_x = rect.left() + zero_ratio * rect.width()
                neg_left = max(left_x, rect.left())
                neg_right = min(zero_x, right_x)
                if neg_right > neg_left:
                    painter.setBrush(QtGui.QColor(COLOR_GAUGE_NEGATIVE))
                    painter.setOpacity(0.18)
                    painter.drawRect(
                        QtCore.QRectF(
                            neg_left, rect.top(), neg_right - neg_left, rect.height()
                        )
                    )
                pos_left = max(zero_x, left_x)
                pos_right = min(right_x, rect.right())
                if pos_right > pos_left:
                    painter.setBrush(QtGui.QColor(COLOR_GAUGE_POSITIVE))
                    painter.setOpacity(0.18)
                    painter.drawRect(
                        QtCore.QRectF(
                            pos_left, rect.top(), pos_right - pos_left, rect.height()
                        )
                    )
                painter.setOpacity(0.6)
                painter.setPen(QtGui.QPen(QtGui.QColor("#888888"), 1))
                painter.drawLine(
                    QtCore.QPointF(zero_x, rect.top() - 2),
                    QtCore.QPointF(zero_x, rect.bottom() + 2),
                )
            else:
                painter.setOpacity(0.18 if not self._inactive else 0.1)
                tint = COLOR_GAUGE_NEGATIVE if max_lim <= 0 else COLOR_GAUGE_POSITIVE
                painter.setBrush(QtGui.QColor(tint))
                painter.drawRect(
                    QtCore.QRectF(left_x, rect.top(), range_width, rect.height())
                )

            if self._inactive or math.isnan(self._angle):
                return

            pointer_value = _clamp(self._angle, GAUGE_GLOBAL_MIN, GAUGE_GLOBAL_MAX)
            pointer_ratio = _clamp(self._map_global(pointer_value), 0.0, 1.0)
            pointer_x = rect.left() + pointer_ratio * rect.width()
            pointer_x = min(max(pointer_x, left_x), right_x)

            if self.out_of_bounds:
                pointer_color = QtGui.QColor(COLOR_GAUGE_EXCEEDED)
            elif self._angle > 0:
                pointer_color = QtGui.QColor(COLOR_GAUGE_POSITIVE).darker(110)
            elif self._angle < 0:
                pointer_color = QtGui.QColor(COLOR_GAUGE_NEGATIVE).darker(110)
            else:
                pointer_color = QtGui.QColor(COLOR_GAUGE_ZERO)

            painter.setOpacity(1.0)
            painter.setPen(QtGui.QPen(pointer_color, 3))
            painter.drawLine(
                QtCore.QPointF(pointer_x, rect.top() - 3),
                QtCore.QPointF(pointer_x, rect.bottom() + 3),
            )

    class JointRowWidget(QtWidgets.QFrame):
        """Displays a single joint row with angle, gauge, and temperatures."""

        def __init__(
            self,
            index: int,
            name: str,
            parent: Optional[QtWidgets.QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self._index = index
            self._blink_active = False
            self._base_color = QtGui.QColor("#888888")
            self._last_t0 = float("nan")

            self.setFrameShape(QtWidgets.QFrame.NoFrame)
            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(12, 6, 12, 6)
            layout.setSpacing(12)

            self._name_label = QtWidgets.QLabel(f"{index:02d}  {name}")
            self._name_label.setFont(make_font(12))
            self._name_label.setStyleSheet("color: #555555;")
            layout.addWidget(self._name_label, 0)

            value_font = make_mono_font(12)
            self._deg_label = QtWidgets.QLabel("--/--")
            self._deg_label.setFont(value_font)
            self._deg_label.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            )
            layout.addWidget(self._deg_label, 0)
            width_hint = (
                self._deg_label.fontMetrics().horizontalAdvance("+180.0°/+3.14") + 12
            )
            self._deg_label.setMinimumWidth(width_hint)

            self._gauge = AngleGauge()
            layout.addWidget(self._gauge, 1)

            temp_layout = QtWidgets.QVBoxLayout()
            temp_layout.setContentsMargins(0, 0, 0, 0)
            temp_layout.setSpacing(0)

            self._temp_label = QtWidgets.QLabel("--°C")
            self._temp_label.setFont(make_mono_font(15, bold=True))
            self._temp_label.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            )
            temp_layout.addWidget(self._temp_label)
            temp_width = (
                self._temp_label.fontMetrics().horizontalAdvance("+999.9°C") + 10
            )
            self._temp_label.setMinimumWidth(temp_width)

            self._temp_aux_label = QtWidgets.QLabel("T₁ --°C")
            self._temp_aux_label.setFont(make_mono_font(11))
            self._temp_aux_label.setStyleSheet("color: #666666;")
            self._temp_aux_label.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            )
            temp_layout.addWidget(self._temp_aux_label)
            aux_width = (
                self._temp_aux_label.fontMetrics().horizontalAdvance("T₁ +999.9°C")
                + 6
            )
            self._temp_aux_label.setMinimumWidth(aux_width)

            layout.addLayout(temp_layout, 0)

        def update_state(self, motor: Any, blink_visible: bool) -> None:
            angle_rad = _safe_float(getattr(motor, "q", float("nan")))
            angle_deg = (
                math.degrees(angle_rad) if not math.isnan(angle_rad) else float("nan")
            )

            temperatures = getattr(motor, "temperature", None)
            t0 = _safe_float(temperatures[0]) if temperatures else float("nan")
            if t0 == 0.0:
                t0 = float("nan")
            t1 = (
                _safe_float(temperatures[1])
                if temperatures and len(temperatures) > 1
                else float("nan")
            )
            if t1 == 0.0:
                t1 = float("nan")

            inactive = math.isnan(t0)
            limits = JOINT_LIMITS.get(self._index)
            self._gauge.update_value(angle_rad, limits)
            self._gauge.set_inactive(inactive or math.isnan(angle_rad))

            if math.isnan(angle_rad) or inactive:
                self._deg_label.setText("--/--")
                self._deg_label.setStyleSheet(
                    "color: #bbbbbb; padding: 1px 6px; border: none;"
                )
            else:
                deg_text = f"{angle_deg:+5.1f}°"
                rad = round(angle_rad, 2)
                rad_text = f"{rad:+.2f}"
                self._deg_label.setText(f"{deg_text}/{rad_text}")
                self._deg_label.setStyleSheet(
                    "color: #303030; padding: 1px 6px; border: none;"
                )

            if inactive:
                self._name_label.setStyleSheet("color: #bbbbbb;")
            else:
                self._name_label.setStyleSheet("color: #555555;")

            self._last_t0 = t0

            if math.isnan(t0):
                self._temp_label.setText("--°C")
                self._base_color = QtGui.QColor("#888888")
            else:
                self._temp_label.setText(f"{t0:5.1f}°C")
                self._base_color = temperature_color(t0)

            if math.isnan(t1):
                self._temp_aux_label.setText("")
            else:
                self._temp_aux_label.setText(f"T₁ {t1:5.1f}°C")

            self._blink_active = not inactive and t0 >= TEMP_BLINK_THRESHOLD
            self._render_blink(blink_visible)

        def clear(self) -> None:
            self._deg_label.setText("--/--")
            self._deg_label.setStyleSheet(
                "color: #bbbbbb; padding: 1px 6px; border: none;"
            )
            limits = JOINT_LIMITS.get(self._index)
            self._gauge.update_value(float("nan"), limits)
            self._gauge.set_inactive(True)
            self._temp_label.setText("--°C")
            self._temp_aux_label.setText("")
            self._base_color = QtGui.QColor("#888888")
            self._name_label.setStyleSheet("color: #bbbbbb;")
            self._blink_active = False
            self._render_blink(True)

        def apply_blink(self, visible: bool) -> None:
            self._render_blink(visible)

        def _render_blink(self, visible: bool) -> None:
            if self._blink_active and visible:
                self._temp_label.setStyleSheet(
                    "color: white; background-color: #d62728; padding: 2px 10px; border-radius: 6px;"
                )
            else:
                color = self._base_color.name()
                self._temp_label.setStyleSheet(
                    f"color: {color}; padding: 2px 4px; border-radius: 6px;"
                )
            if not self._temp_aux_label.text():
                self._temp_aux_label.setStyleSheet("color: #bbbbbb;")
            else:
                self._temp_aux_label.setStyleSheet("color: #666666;")

    class JointGroupWidget(QtWidgets.QFrame):
        """Card-like container for a group of joints."""

        def __init__(
            self,
            title: str,
            joints: Sequence[Tuple[str, int]],
            parent: Optional[QtWidgets.QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self._joints = list(joints)
            self._rows: Dict[int, JointRowWidget] = {}

            self.setObjectName("JointGroupCard")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(10)

            header = QtWidgets.QLabel(title)
            header.setFont(make_font(20, bold=True, emoji=True))
            header.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            layout.addWidget(header)

            for name, index in self._joints:
                row = JointRowWidget(index, name, self)
                layout.addWidget(row)
                self._rows[index] = row

        def update_state(self, state: LowState_, blink_visible: bool) -> None:
            for _, index in self._joints:
                motor = state.motor_state[index]
                self._rows[index].update_state(motor, blink_visible)

        def clear(self) -> None:
            for row in self._rows.values():
                row.clear()

        def apply_blink(self, visible: bool) -> None:
            for row in self._rows.values():
                row.apply_blink(visible)

    class BatteryPanel(QtWidgets.QFrame):
        """Battery telemetry card."""

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self.setObjectName("BatteryCard")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(10)

            header = QtWidgets.QLabel("🔋 Battery")
            header.setFont(make_font(20, bold=True, emoji=True))
            layout.addWidget(header)

            grid = QtWidgets.QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(6)
            layout.addLayout(grid)

            labels = [
                ("Voltage", "_voltage_label"),
                ("Current", "_current_label"),
                ("Temperature", "_temperature_label"),
                ("Cycle", "_cycle_label"),
            ]
            for row, (name, attr) in enumerate(labels):
                label = QtWidgets.QLabel(name)
                label.setFont(make_font(12, bold=True))
                grid.addWidget(label, row, 0, QtCore.Qt.AlignRight)
                value = QtWidgets.QLabel("--")
                value.setFont(make_mono_font(14, bold=True))
                grid.addWidget(value, row, 1, QtCore.Qt.AlignLeft)
                setattr(self, attr, value)

            self._status_label = QtWidgets.QLabel("Battery telemetry")
            self._status_label.setFont(make_font(11))
            self._status_label.setStyleSheet("color: #202020;")
            layout.addWidget(self._status_label)

            self._has_bms = False

        def show_unavailable(self, message: str) -> None:
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: #a33f3f;")

        def update_from_bms(self, bms: BmsState_) -> None:
            self._has_bms = True
            voltage = getattr(bms, "voltage", None)
            current = getattr(bms, "current", None)
            temp = getattr(bms, "temperature", None)
            cycle = getattr(bms, "cycle", None)
            soh = getattr(bms, "soh", None)

            temp_value = None
            if temp is not None and len(temp) > 0:
                temp_value = max(temp)

            self._voltage_label.setText(
                "-- V" if voltage is None else f"{voltage:5.2f} V"
            )
            self._current_label.setText(
                "-- A" if current is None else f"{current:5.2f} A"
            )
            self._temperature_label.setText(
                "-- °C" if temp_value is None else f"{temp_value:4.1f} °C"
            )
            self._cycle_label.setText("--" if cycle is None else f"{cycle:d}")

            status_bits = getattr(bms, "fn", None)
            soh_text = f"SOH {soh}%" if soh is not None else ""
            status_parts = [
                part
                for part in (
                    f"FN {status_bits}" if status_bits is not None else "",
                    soh_text,
                )
                if part
            ]
            status = (
                " | ".join(status_parts) if status_parts else "Battery telemetry"
            )
            self._status_label.setText(status)
            self._status_label.setStyleSheet("color: #202020;")

        def update_from_lowstate(self, state: LowState_) -> None:
            if self._has_bms:
                return
            voltage = getattr(state, "power_v", None)
            current = getattr(state, "power_a", None)
            ntc1 = getattr(state, "temperature_ntc1", None)
            ntc2 = getattr(state, "temperature_ntc2", None)
            temps = [float(ntc) for ntc in (ntc1, ntc2) if ntc not in (None, 0)]
            temp_value = max(temps) if temps else None

            if voltage is not None and not math.isnan(voltage):
                self._voltage_label.setText(f"{voltage:5.2f} V")
            else:
                self._voltage_label.setText("-- V")

            if current is not None and not math.isnan(current):
                self._current_label.setText(f"{current:5.2f} A")
            else:
                self._current_label.setText("-- A")

            if temp_value is not None:
                self._temperature_label.setText(f"{temp_value:4.1f} °C")
            else:
                self._temperature_label.setText("-- °C")

            if not self._has_bms:
                self._status_label.setText("Derived from low-state")
                self._status_label.setStyleSheet("color: #777777;")

        @property
        def has_bms(self) -> bool:
            return self._has_bms

    class CameraPanel(QtWidgets.QFrame):
        """Embedded live camera preview card with manual activation."""

        _INTERVAL_MS = 250
        _DISPLAY_SIZE = QtCore.QSize(360, 240)

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self.setObjectName("CameraCard")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(20, 18, 20, 18)
            layout.setSpacing(12)

            header = QtWidgets.QLabel("🎥 Camera")
            header.setFont(make_font(20, bold=True, emoji=True))
            layout.addWidget(header)

            self._view = QtWidgets.QLabel("Camera idle. Start the feed when needed.")
            self._view.setAlignment(QtCore.Qt.AlignCenter)
            self._view.setMinimumSize(self._DISPLAY_SIZE)
            self._view.setMaximumHeight(self._DISPLAY_SIZE.height())
            self._view.setStyleSheet(
                "background-color: #111; color: #d0d0d0; border-radius: 10px;"
            )
            layout.addWidget(self._view, 1)

            self._button = QtWidgets.QPushButton("Start Live View")
            self._button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self._button.setCheckable(True)
            self._button.setStyleSheet(
                """
                QPushButton {
                    background-color: #2d7ff9;
                    color: white;
                    padding: 6px 16px;
                    border-radius: 10px;
                    font-weight: 600;
                }
                QPushButton:checked {
                    background-color: #d64545;
                }
                QPushButton:disabled {
                    background-color: #666666;
                    color: #cccccc;
                }
                """
            )
            layout.addWidget(self._button, alignment=QtCore.Qt.AlignRight)

            self._client = None
            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(self._INTERVAL_MS)
            self._timer.timeout.connect(self._update_frame)
            self._failure_streak = 0
            self._last_pixmap: Optional[QtGui.QPixmap] = None
            self._active = False

            if VideoClient is None:
                self._button.setEnabled(False)
                self._view.setText("Video client unavailable on this system.")
            else:
                self._button.toggled.connect(self._toggle_stream)

        def _toggle_stream(self, enabled: bool) -> None:
            if enabled:
                if self._client is None:
                    try:
                        self._client = create_video_client()
                    except Exception as exc:  # pragma: no cover - depends on HW
                        self._button.blockSignals(True)
                        self._button.setChecked(False)
                        self._button.blockSignals(False)
                        self._view.setText(f"Camera unavailable: {exc}")
                        self._client = None
                        return
                self._failure_streak = 0
                self._view.setText("Starting camera stream…")
                self._timer.start()
                self._active = True
                self._button.setText("Stop Live View")
            else:
                self._timer.stop()
                self._active = False
                if self._last_pixmap is not None:
                    self._view.setPixmap(self._last_pixmap)
                    self._view.setText("")
                else:
                    self._view.setText("Camera idle. Start the feed when needed.")
                    self._view.setPixmap(QtGui.QPixmap())
                self._button.setText("Start Live View")

        def _update_frame(self) -> None:
            if not self._active or self._client is None:
                return
            try:
                code, data = self._client.GetImageSample()
            except Exception as exc:  # pragma: no cover - depends on HW
                self._handle_failure(f"Camera error: {exc}")
                return
            if code != 0 or not data:
                self._handle_failure(f"Camera unavailable (code={code})")
                return

            image = QtGui.QImage.fromData(bytes(data))
            if image.isNull():
                self._handle_failure("Failed to decode camera frame")
                return

            pixmap = QtGui.QPixmap.fromImage(image)
            if pixmap.isNull():
                self._handle_failure("Failed to convert camera frame")
                return

            self._failure_streak = 0
            scaled = pixmap.scaled(
                self._DISPLAY_SIZE,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
            self._last_pixmap = scaled
            self._view.setText("")
            self._view.setPixmap(scaled)

        def _handle_failure(self, message: str) -> None:
            self._failure_streak += 1
            if self._failure_streak >= 5:
                self._view.setText(message)
                self._view.setPixmap(QtGui.QPixmap())

        def shutdown(self) -> None:
            if self._timer.isActive():
                self._timer.stop()
            if self._button.isChecked():
                self._button.blockSignals(True)
                self._button.setChecked(False)
                self._button.blockSignals(False)
            self._active = False
            if self._last_pixmap is not None:
                self._view.setPixmap(self._last_pixmap)
                self._view.setText("")
            else:
                self._view.setText("Camera idle. Start the feed when needed.")
                self._view.setPixmap(QtGui.QPixmap())

    class DepthPanel(QtWidgets.QFrame):
        """Placeholder depth view card; idle until data is available."""

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self.setObjectName("DepthCard")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(20, 18, 20, 18)
            layout.setSpacing(12)

            header = QtWidgets.QLabel("🌊 Depth")
            header.setFont(make_font(20, bold=True, emoji=True))
            layout.addWidget(header)

            self._view = QtWidgets.QLabel(
                "Depth feed disabled. Toggle when a source is ready."
            )
            self._view.setAlignment(QtCore.Qt.AlignCenter)
            self._view.setMinimumSize(CameraPanel._DISPLAY_SIZE)
            self._view.setMaximumHeight(CameraPanel._DISPLAY_SIZE.height())
            self._view.setStyleSheet(
                "background-color: #202225; color: #d0d0d0; border-radius: 10px;"
            )
            layout.addWidget(self._view, 1)

        def shutdown(self) -> None:
            pass

    class JointStateMonitorWindow(QtWidgets.QMainWindow):
        """Main application window for monitoring."""

        def __init__(
            self,
            receiver: StateReceiver,
            parent: Optional[QtWidgets.QWidget] = None,
        ):
            super().__init__(parent)
            self._receiver = receiver
            self._blink_visible = True

            self.setWindowTitle("Unitree G1 Joint Monitor")
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                geometry = screen.availableGeometry()
                self.setGeometry(geometry)
            self.showMaximized()

            base_font = make_font(14)
            self.setFont(base_font)

            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #f3f4f7;
                }
                QFrame#JointGroupCard {
                    background-color: #ffffff;
                    border: 1px solid #d9dde2;
                    border-radius: 14px;
                }
                QFrame#BatteryCard {
                    background-color: #ffffff;
                    border: 1px solid #d9dde2;
                    border-radius: 14px;
                }
                QFrame#CameraCard {
                    background-color: #ffffff;
                    border: 1px solid #d9dde2;
                    border-radius: 14px;
                }
                QFrame#DepthCard {
                    background-color: #ffffff;
                    border: 1px solid #d9dde2;
                    border-radius: 14px;
                }
                """
            )

            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            main_layout = QtWidgets.QVBoxLayout(central)
            main_layout.setContentsMargins(20, 18, 20, 18)
            main_layout.setSpacing(16)

            scroll_area = QtWidgets.QScrollArea(self)
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            main_layout.addWidget(scroll_area, stretch=1)

            scroller_content = QtWidgets.QWidget()
            scroll_area.setWidget(scroller_content)
            grid = QtWidgets.QGridLayout(scroller_content)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(18)
            for col in range(4):
                grid.setColumnStretch(col, 1)

            self._groups: Dict[str, JointGroupWidget] = {}
            self._battery_card = BatteryPanel(self)
            grid.addWidget(self._battery_card, 0, 3)

            def add_group(
                row: int, column: int, title: str, span: Tuple[int, int] = (1, 1)
            ) -> None:
                group = JointGroupWidget(title, JOINT_GROUPS[title], self)
                self._groups[title] = group
                grid.addWidget(group, row, column, *span)

            add_group(0, 0, "🦵 Left Leg")
            add_group(0, 1, "🦵 Right Leg")
            add_group(0, 2, "🧍 Waist")
            add_group(1, 0, "🫲 Left Arm")
            add_group(1, 1, "🫱 Right Arm")

            self._camera_panel = CameraPanel(self)
            grid.addWidget(self._camera_panel, 1, 2)
            self._depth_panel = DepthPanel(self)
            grid.addWidget(self._depth_panel, 1, 3)

            self._battery_receiver = None
            try:
                self._battery_receiver = BatteryReceiver()
            except RuntimeError as exc:
                self._battery_card.show_unavailable(str(exc))
            else:
                if not self._battery_receiver.active:
                    message = (
                        self._battery_receiver.error
                        or "Battery telemetry unavailable"
                    )
                    self._battery_card.show_unavailable(message)

            self._update_timer = QtCore.QTimer(self)
            self._update_timer.timeout.connect(self._refresh_from_state)
            self._update_timer.start(100)

            self._blink_timer = QtCore.QTimer(self)
            self._blink_timer.timeout.connect(self._toggle_blink)
            self._blink_timer.start(500)

        def _toggle_blink(self) -> None:
            self._blink_visible = not self._blink_visible
            for group in self._groups.values():
                group.apply_blink(self._blink_visible)

        def _refresh_from_state(self) -> None:
            state = self._receiver.consume_latest()
            if state is None:
                return

            for group in self._groups.values():
                group.update_state(state, self._blink_visible)

            bms_msg = None
            if self._battery_receiver and self._battery_receiver.active:
                bms_msg = self._battery_receiver.consume_latest()
            if bms_msg is not None:
                self._battery_card.update_from_bms(bms_msg)
            elif not self._battery_card.has_bms:
                self._battery_card.update_from_lowstate(state)

        def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
            if event.key() == QtCore.Qt.Key_Escape:
                self.close()
            else:
                super().keyPressEvent(event)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
            if hasattr(self, "_camera_panel"):
                self._camera_panel.shutdown()
            if hasattr(self, "_depth_panel"):
                self._depth_panel.shutdown()
            super().closeEvent(event)

else:  # pragma: no cover - fallback when PyQt5 is absent.

    def make_font(*_: object, **__: object) -> None:
        raise RuntimeError("make_font requires PyQt5.")

    def make_mono_font(*_: object, **__: object) -> None:
        raise RuntimeError("make_mono_font requires PyQt5.")

    class CameraWidget:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("CameraWidget requires PyQt5 but it is not installed.")

    class AngleGauge:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("AngleGauge requires PyQt5.")

    class JointRowWidget:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("JointRowWidget requires PyQt5.")

    class JointGroupWidget:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("JointGroupWidget requires PyQt5.")

    class BatteryPanel:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("BatteryPanel requires PyQt5.")

    class CameraPanel:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("CameraPanel requires PyQt5.")

    class DepthPanel:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("DepthPanel requires PyQt5.")

    class JointStateMonitorWindow:  # type: ignore[override]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("JointStateMonitorWindow requires PyQt5.")


__all__ = [
    "_require_pyqt",
    "camera_frame_interval_ms",
    "_ensure_cv2",
    "_clamp",
    "_safe_float",
    "temperature_color",
    "make_font",
    "make_mono_font",
    "CameraWidget",
    "StateReceiver",
    "BatteryReceiver",
    "AngleGauge",
    "JointRowWidget",
    "JointGroupWidget",
    "BatteryPanel",
    "CameraPanel",
    "DepthPanel",
    "JointStateMonitorWindow",
]

