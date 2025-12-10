#!/usr/bin/env python3
"""
PyQt-based control panel for the Unitree G1 locomotion interface.

The UI presents the keyboard legend, mirrors keyboard shortcuts as buttons,
allows sending custom FSM IDs with confirmation, and keeps a live status readout
of the robot's state.  Keyboard and button interactions are funneled through
``TeleopController`` from ``keyboard_teleop.py``.
"""

from __future__ import annotations

import argparse
import sys
import math
from typing import Dict, Optional, Sequence

from const import DEFAULT_INTERFACE, MODE_HOLD_DURATION, JOINT_GROUPS
from keyboard_teleop import (
    DEFAULT_BINDINGS,
    TeleopController,
    format_key_label,
)
from utilities import ensure_channel_factory, create_video_client
from gui_widgets import (
    JointGroupWidget,
    BatteryPanel,
    BatteryReceiver,
    StateReceiver,
    CameraPanel,
)

try:
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
except ImportError as exc:  # pragma: no cover - audio client optional
    AudioClient = None  # type: ignore[assignment]
    AUDIO_IMPORT_ERROR = exc
else:
    AUDIO_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Qt imports with PyQt5 / PyQt6 compatibility
# ---------------------------------------------------------------------------

try:  # Prefer PyQt5 but allow PyQt6 as a fallback.
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

    QT_VERSION = 5
    Qt = QtCore.Qt  # type: ignore[assignment]
except ImportError:  # pragma: no cover - depends on environment.
    from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore

    QT_VERSION = 6
    Qt = QtCore.Qt  # type: ignore[assignment]


DEFAULT_TTS_VOICE = 1
DEFAULT_TTS_VOLUME = 80
DEFAULT_TTS_TIMEOUT = 10.0

SHORT_LABELS: Dict[str, str] = {
    "forward": "Forward",
    "backward": "Reverse",
    "left": "Strafe Left",
    "right": "Strafe Right",
    "turn_left": "Turn Left",
    "turn_right": "Turn Right",
    "stop": "Hold Still",
    "handshake": "Handshake",
    "mode_zero_torque": "Zero Torque",
    "mode_locked": "Balance Stand",
    "mode_sit": "Sit",
    "mode_standup": "Stand Assist",
    "mode_squat": "Squat",
    "mode_squat_recover": "Squat→Stand",
    "mode_lie_recover": "Lie→Stand",
    "mode_regular_walk": "Walk",
    "mode_run": "Run",
    "damp": "Damp (Confirm)",
    "start": "Start Balance",
    "quit": "Exit Tele-op",
}

ADVANCED_ACTIONS = {"mode_locked"}
SUPPORT_ACTIONS = {"start"}

def _qt_key(name: str) -> int:
    """Return the Qt key code for ``name`` across PyQt5/6."""
    if QT_VERSION == 5:
        return getattr(Qt, name)
    return getattr(Qt.Key, name)  # type: ignore[attr-defined]


def _align_hcenter() -> int:
    return (
        Qt.AlignHCenter  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.AlignmentFlag.AlignHCenter  # type: ignore[attr-defined]
    )


def _align_left() -> int:
    return (
        Qt.AlignLeft  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.AlignmentFlag.AlignLeft  # type: ignore[attr-defined]
    )


def _focus_policy_no_focus():
    return (
        Qt.NoFocus  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.FocusPolicy.NoFocus  # type: ignore[attr-defined]
    )


def _focus_policy_strong():
    return (
        Qt.StrongFocus  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.FocusPolicy.StrongFocus  # type: ignore[attr-defined]
    )


def _toolbutton_text_beside_icon():
    return (
        Qt.ToolButtonTextBesideIcon  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.ToolButtonStyle.ToolButtonTextBesideIcon  # type: ignore[attr-defined]
    )


def _scrollbar_always_off():
    return (
        Qt.ScrollBarAlwaysOff  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff  # type: ignore[attr-defined]
    )


def _scrollbar_as_needed():
    return (
        Qt.ScrollBarAsNeeded  # type: ignore[attr-defined]
        if QT_VERSION == 5
        else Qt.ScrollBarPolicy.ScrollBarAsNeeded  # type: ignore[attr-defined]
    )


APP_STYLESHEET = """
QMainWindow {
    background-color: #f6f7fb;
    color: #1a202c;
}
QLabel#headerLabel {
    font-size: 22px;
    font-weight: 600;
    color: #21507f;
}
QLabel#statusLabel {
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid #d5dbe5;
    background-color: #ffffff;
}
QLabel#eventLabel {
    font-size: 13px;
    padding: 10px 12px;
    border-radius: 8px;
    background-color: #fff4de;
    color: #8a4f14;
    border: 1px solid #f3d09c;
}
QLabel#holdsLabel {
    font-style: italic;
    color: #4a5a73;
}
QGroupBox {
    border: 1px solid #d5dbe5;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 4px;
    color: #21507f;
    font-weight: 600;
}
QPushButton {
    background-color: #ffffff;
    color: #1a202c;
    border-radius: 6px;
    padding: 8px 14px;
    border: 1px solid #c8d2e1;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #e7f1ff;
}
QPushButton:pressed {
    background-color: #d3e4ff;
}
QLineEdit {
    border: 1px solid #cfd6e3;
    border-radius: 6px;
    padding: 6px 8px;
    background-color: #ffffff;
    color: #1a202c;
    selection-background-color: #2f81f7;
}
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f2f5fb;
    border: 1px solid #d5dbe5;
    border-radius: 6px;
    gridline-color: #e2e6ef;
    selection-background-color: #cfe1ff;
    selection-color: #1a202c;
    color: #1a202c;
    font-size: 12px;
}
QHeaderView::section {
    background-color: #eef2fa;
    color: #53627a;
    padding: 6px;
    border: none;
    font-weight: 600;
}
QToolButton#CollapsibleToggle {
    border: none;
    font-weight: 600;
    color: #21507f;
    padding: 4px 2px;
}
"""


class CollapsibleSection(QtWidgets.QWidget):
    """Reusable collapsible container with a toggle header."""

    toggled = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        content: QtWidgets.QWidget,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        start_collapsed: bool = True,
    ) -> None:
        super().__init__(parent)
        self._content = content

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._toggle = QtWidgets.QToolButton()
        self._toggle.setObjectName("CollapsibleToggle")
        self._toggle.setText(title)
        self._toggle.setToolButtonStyle(_toolbutton_text_beside_icon())
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not start_collapsed)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if not start_collapsed else Qt.ArrowType.RightArrow
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle, 0, _align_left())

        layout.addWidget(content)
        content.setVisible(not start_collapsed)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)
        self.toggled.emit(checked)


class TtsInterface:
    """Wrapper around the Unitree audio client for quick TTS playback."""

    def __init__(
        self,
        interface: str,
        *,
        default_voice: int = DEFAULT_TTS_VOICE,
        default_volume: int = DEFAULT_TTS_VOLUME,
        timeout: float = DEFAULT_TTS_TIMEOUT,
    ) -> None:
        self._interface = interface
        self._default_voice = int(default_voice)
        self._volume = int(default_volume)
        self._timeout = float(timeout)
        self._client: Optional[AudioClient] = None
        self._available = AudioClient is not None
        self._error_message = (
            str(AUDIO_IMPORT_ERROR) if AUDIO_IMPORT_ERROR is not None else ""
        )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error_message(self) -> str:
        if self._available:
            return ""
        return self._error_message or "Audio client is not available."

    def speak(self, message: str) -> None:
        if not self._available:
            raise RuntimeError(self.error_message)
        message = message.strip()
        if not message:
            raise ValueError("TTS message must not be empty.")

        client = self._ensure_client()

        code = client.TtsMaker(message, self._default_voice)
        if code != 0:
            raise RuntimeError(f"TTS call failed with error code {code}.")

    def close(self) -> None:
        self._client = None

    def _ensure_client(self) -> AudioClient:
        if not self._available or AudioClient is None:
            raise RuntimeError(self.error_message or "Audio client unavailable.")
        if self._client is not None:
            return self._client

        ensure_channel_factory(self._interface)
        client = AudioClient()
        try:
            client.SetTimeout(self._timeout)
        except AttributeError:
            pass
        client.Init()
        if self._volume:
            client.SetVolume(self._volume)
        self._client = client
        return client


class JointTelemetryPanel(QtWidgets.QWidget):
    """Embedded joint telemetry visualiser mirroring the monitor window."""

    def __init__(
        self,
        interface: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._receiver: Optional[StateReceiver] = None
        self._battery_receiver: Optional[BatteryReceiver] = None
        self._groups: Dict[str, JointGroupWidget] = {}
        self._blink_visible = True
        self._available = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._status_label = QtWidgets.QLabel("Expand to start streaming joint telemetry.")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(320)
        self._scroll.setHorizontalScrollBarPolicy(_scrollbar_always_off())
        self._scroll.setVerticalScrollBarPolicy(_scrollbar_as_needed())
        layout.addWidget(self._scroll, 1)

        self._content = QtWidgets.QWidget()
        self._scroll.setWidget(self._content)

        self._grid = QtWidgets.QGridLayout(self._content)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(12)

        try:
            ensure_channel_factory(interface)
            self._receiver = StateReceiver()
        except Exception as exc:  # pragma: no cover - hardware / SDK dependency
            self._available = False
            self._status_label.setText(f"Joint telemetry unavailable: {exc}")
            self._scroll.setVisible(False)
            return

        for row, (title, joints) in enumerate(JOINT_GROUPS.items()):
            widget = JointGroupWidget(title, joints, self._content)
            self._grid.addWidget(widget, row // 3, row % 3)
            self._groups[title] = widget

        self._battery_panel = BatteryPanel(self._content)
        self._grid.addWidget(self._battery_panel, 0, 3, 2, 1)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(2, 1)
        self._grid.setColumnStretch(3, 1)

        self._reset_visuals()

        try:
            self._battery_receiver = BatteryReceiver()
            if not self._battery_receiver.active:
                message = (
                    self._battery_receiver.error
                    if hasattr(self._battery_receiver, "error")
                    else "Battery telemetry unavailable."
                )
                self._battery_panel.show_unavailable(message or "Battery telemetry unavailable.")
        except Exception as exc:  # pragma: no cover - depends on SDK
            self._battery_panel.show_unavailable(str(exc))
            self._battery_receiver = None

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self._refresh_from_state)

        self._blink_timer = QtCore.QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)

    @property
    def available(self) -> bool:
        return self._available

    def set_active(self, active: bool) -> None:
        if not self._available:
            return
        if active:
            self._status_label.setText("Streaming joint telemetry…")
            self._update_timer.start(100)
            self._blink_timer.start(500)
        else:
            self._status_label.setText("Telemetry paused (panel collapsed).")
            self._update_timer.stop()
            self._blink_timer.stop()
            self._reset_visuals()

    def shutdown(self) -> None:
        if not self._available:
            return
        self._update_timer.stop()
        self._blink_timer.stop()

    def _reset_visuals(self) -> None:
        for widget in self._groups.values():
            widget.clear()

        try:
            self._battery_panel.clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def _toggle_blink(self) -> None:
        self._blink_visible = not self._blink_visible
        for group in self._groups.values():
            group.apply_blink(self._blink_visible)

    def _refresh_from_state(self) -> None:
        if self._receiver is None:
            return

        state = self._receiver.consume_latest()
        if state is None:
            return

        for widget in self._groups.values():
            widget.update_state(state, self._blink_visible)

        battery_panel = getattr(self, "_battery_panel", None)
        if battery_panel is None:
            return

        if self._battery_receiver and getattr(self._battery_receiver, "active", False):
            bms = self._battery_receiver.consume_latest()
            if bms is not None:
                battery_panel.update_from_bms(bms)
                return

        try:
            battery_panel.update_from_lowstate(state)
        except AttributeError:
            pass


class _InlineCameraPanel(CameraPanel):
    """Camera panel variant that honours the selected network interface."""

    _DISPLAY_SIZE = QtCore.QSize(640, 400)

    def __init__(self, interface: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._interface = interface
        ensure_channel_factory(interface)
        super().__init__(parent)

    def _create_video_client(self) -> Optional[object]:
        try:
            return create_video_client(self._interface)
        except Exception as exc:  # pragma: no cover - depends on HW
            return exc

    def _toggle_stream(self, enabled: bool) -> None:  # type: ignore[override]
        if enabled:
            if self._client is None:
                result = self._create_video_client()
                if isinstance(result, Exception):
                    self._button.blockSignals(True)
                    self._button.setChecked(False)
                    self._button.blockSignals(False)
                    self._view.setText(f"Camera unavailable: {result}")
                    self._client = None
                    return
                self._client = result
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


class CameraPanelWrapper(QtWidgets.QWidget):
    """Wrapper around the optional camera panel so it fits the collapsible layout."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        interface: str,
    ) -> None:
        super().__init__(parent)
        self._camera: Optional[_InlineCameraPanel] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        try:
            panel = _InlineCameraPanel(interface, self)
        except Exception as exc:  # pragma: no cover - depends on SDK
            fallback = QtWidgets.QLabel(f"Camera panel unavailable: {exc}")
            fallback.setWordWrap(True)
            fallback.setAlignment(_align_left())
            layout.addWidget(fallback)
            self._camera = None
        else:
            panel.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
            panel.setMinimumHeight(panel.minimumSizeHint().height())
            layout.addWidget(panel)
            self._camera = panel

    def shutdown(self) -> None:
        if self._camera is not None:
            try:
                self._camera.shutdown()
            except Exception:
                pass

    def _toggle_blink(self) -> None:
        self._blink_visible = not self._blink_visible
        for group in self._groups.values():
            group.apply_blink(self._blink_visible)

    def _refresh_from_state(self) -> None:
        if self._receiver is None:
            return
        state = self._receiver.consume_latest()
        if state is None:
            return

        for widget in self._groups.values():
            widget.update_state(state, self._blink_visible)

        battery_panel = getattr(self, "_battery_panel", None)
        if battery_panel is None:
            return

        if self._battery_receiver and getattr(self._battery_receiver, "active", False):
            bms = self._battery_receiver.consume_latest()
            if bms is not None:
                battery_panel.update_from_bms(bms)
                return

        try:
            battery_panel.update_from_lowstate(state)
        except AttributeError:
            pass

class KeyEventFilter(QtCore.QObject):
    """Global event filter that routes keyboard events to the main window."""

    def __init__(self, window: "ControlPanelWindow") -> None:
        super().__init__(window)
        self._window = window

    def eventFilter(self, obj, event):  # type: ignore[override]
        etype = event.type()
        if etype == QtCore.QEvent.Type.KeyPress:
            if self._window._handle_key_event(event, is_press=True):
                return True
        elif etype == QtCore.QEvent.Type.KeyRelease:
            if self._window._handle_key_event(event, is_press=False):
                return True
        return super().eventFilter(obj, event)


class ControlPanelWindow(QtWidgets.QMainWindow):
    """Main application window that wires the PyQt UI to TeleopController."""

    def __init__(
        self,
        *,
        interface: str = DEFAULT_INTERFACE,
        bindings: Sequence = DEFAULT_BINDINGS,
    ) -> None:
        super().__init__()
        self.setWindowTitle("ELLISa G1 Control Panel")
        self.setMinimumSize(860, 620)

        self._controller = TeleopController(interface=interface, bindings=bindings)
        self._tts_interface = TtsInterface(interface)

        self._char_action_map: Dict[str, str] = {}
        self._special_action_map: Dict[int, str] = {}
        self._build_key_lookup(bindings)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(_scrollbar_always_off())
        scroll.setVerticalScrollBarPolicy(_scrollbar_as_needed())
        self.setCentralWidget(scroll)

        content = QtWidgets.QWidget(scroll)
        content.setFocusPolicy(_focus_policy_strong())
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(18)
        layout.setContentsMargins(24, 24, 24, 24)

        header = QtWidgets.QLabel("ELLISa G1 Control Panel")
        header.setObjectName("headerLabel")
        header.setAlignment(_align_hcenter())
        layout.addWidget(header)

        self._status_label = QtWidgets.QLabel()
        self._status_label.setObjectName("statusLabel")
        layout.addWidget(self._status_label)

        self._holds_label = QtWidgets.QLabel()
        self._holds_label.setObjectName("holdsLabel")
        layout.addWidget(self._holds_label)

        self._event_label = QtWidgets.QLabel()
        self._event_label.setObjectName("eventLabel")
        self._event_label.setWordWrap(True)
        layout.addWidget(self._event_label)

        layout.addWidget(self._build_button_panel(bindings))

        advanced_section = self._build_advanced_panel(bindings)
        if advanced_section is not None:
            layout.addWidget(advanced_section)

        layout.addWidget(self._build_custom_command_panel())
        layout.addWidget(self._build_tts_panel())

        support_panel = self._build_support_panel(bindings)
        if support_panel is not None:
            layout.addWidget(support_panel)

        self._joint_panel = JointTelemetryPanel(interface, self)
        self._joint_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self._joint_panel.setMinimumHeight(420)
        joint_title = (
            "Joint Telemetry"
            if self._joint_panel.available
            else "Joint Telemetry (unavailable)"
        )
        start_collapsed = True
        self._joint_section = CollapsibleSection(
            joint_title,
            self._joint_panel,
            self,
            start_collapsed=start_collapsed,
        )
        self._joint_section.toggled.connect(self._joint_panel.set_active)
        self._joint_panel.set_active(False)
        layout.addWidget(self._joint_section)

        self._camera_panel = CameraPanelWrapper(self, interface=interface)
        camera_title = "Camera View"
        self._camera_section = CollapsibleSection(
            camera_title,
            self._camera_panel,
            self,
            start_collapsed=True,
        )
        layout.addWidget(self._camera_section)
        self._camera_section.toggled.connect(self._on_camera_section_toggled)

        layout.addWidget(self._build_legend_panel(bindings))
        layout.addStretch(1)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(50)

        self._state_timer = QtCore.QTimer(self)
        self._state_timer.timeout.connect(self._controller.refresh_state)
        self._state_timer.start(500)

        self._update_status_labels()
        self._update_active_holds_label()
        self._update_tts_status("")

        self._key_filter = KeyEventFilter(self)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self._key_filter)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _build_button_panel(self, bindings: Sequence) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )

        mode_actions = [
            b
            for b in bindings
            if b.category in {"Modes", "Recovery"}
            and b.behavior == "press"
            and b.action not in ADVANCED_ACTIONS
        ]
        handshake_binding = next((b for b in bindings if b.action == "handshake"), None)
        if handshake_binding is not None and handshake_binding not in mode_actions:
            mode_actions.append(handshake_binding)

        if mode_actions:
            layout.addWidget(
                self._create_button_group("", mode_actions), 2
            )

        layout.addStretch(1)

        return container

    def _build_advanced_panel(
        self, bindings: Sequence
    ) -> Optional[CollapsibleSection]:
        advanced_bindings = [
            binding for binding in bindings if binding.action in ADVANCED_ACTIONS
        ]
        if not advanced_bindings:
            return None

        content = QtWidgets.QWidget(self)
        vlayout = QtWidgets.QVBoxLayout(content)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(8)
        vlayout.addWidget(
            self._create_button_group("Balance & Safety Modes", advanced_bindings)
        )
        content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        return CollapsibleSection(
            "Occasional Modes",
            content,
            self,
            start_collapsed=True,
        )

    def _create_button_group(
        self, title: str, actions: Sequence
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QFrame()
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        columns = 2
        for index, binding in enumerate(actions):
            label = self._display_label(binding)
            button = QtWidgets.QPushButton(label)
            button.setToolTip(
                f"{binding.description}\nKeyboard: {format_key_label(binding.key)}"
            )
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
            button.setMinimumSize(150, 40)
            button.clicked.connect(
                lambda _checked=False, action=binding.action: self._on_action_button(
                    action
                )
            )
            row = index // columns
            col = index % columns
            grid.addWidget(button, row, col)
        return group

    def _display_label(self, binding) -> str:
        label = SHORT_LABELS.get(binding.action)
        if label:
            return label
        description = binding.description.strip()
        if "(" in description:
            description = description.split("(", 1)[0].strip()
        return description or binding.action.replace("_", " ").title()

    def _build_custom_command_panel(self) -> QtWidgets.QFrame:
        group = QtWidgets.QFrame()
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        layout = QtWidgets.QHBoxLayout(group)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        label = QtWidgets.QLabel("FSM ID:")
        layout.addWidget(label)

        self._fsm_input = QtWidgets.QLineEdit()
        self._fsm_input.setPlaceholderText("Enter FSM ID (decimal or 0x hex)")
        self._fsm_input.returnPressed.connect(self._on_submit_custom_fsm)
        self._fsm_input.setMinimumWidth(160)
        self._fsm_input.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        layout.addWidget(self._fsm_input, 1)

        send_button = QtWidgets.QPushButton("Send")
        send_button.clicked.connect(self._on_submit_custom_fsm)
        layout.addWidget(send_button)
        layout.addStretch(1)

        return group

    def _build_tts_panel(self) -> QtWidgets.QFrame:
        group = QtWidgets.QFrame()
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        prompt = QtWidgets.QLabel("Message:")
        row.addWidget(prompt, 0)

        self._tts_input = QtWidgets.QLineEdit()
        self._tts_input.setPlaceholderText("Type what the robot should say…")
        self._tts_input.returnPressed.connect(self._on_send_tts)
        self._tts_input.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        row.addWidget(self._tts_input, 1)

        send_button = QtWidgets.QPushButton("Speak")
        send_button.clicked.connect(self._on_send_tts)
        row.addWidget(send_button, 0)
        layout.addLayout(row)

        self._tts_status_label = QtWidgets.QLabel(
            "Voice 1 • Volume 80% (defaults)"
        )
        self._tts_status_label.setStyleSheet("color: #4a5a73;")
        self._tts_status_label.setWordWrap(True)
        layout.addWidget(self._tts_status_label)

        if not self._tts_interface.available:
            message = self._tts_interface.error_message or "Audio client unavailable."
            self._tts_input.setEnabled(False)
            send_button.setEnabled(False)
            self._tts_status_label.setStyleSheet("color: #b55656;")
            self._tts_status_label.setText(message)

        return group

    def _build_support_panel(self, bindings: Sequence) -> Optional[QtWidgets.QWidget]:
        support_actions = [
            b for b in bindings if b.action in SUPPORT_ACTIONS and b.behavior == "press"
        ]
        if not support_actions:
            return None

        frame = QtWidgets.QFrame()
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        for binding in support_actions:
            button = QtWidgets.QPushButton(self._display_label(binding))
            button.setToolTip(
                f"{binding.description}\nKeyboard: {format_key_label(binding.key)}"
            )
            button.setMinimumSize(150, 40)
            button.clicked.connect(
                lambda _checked=False, action=binding.action: self._on_action_button(
                    action
                )
            )
            layout.addWidget(button, 0)

        layout.addStretch(1)
        return frame

    def _build_legend_panel(self, bindings: Sequence) -> QtWidgets.QFrame:
        group = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        table = QtWidgets.QTableWidget(len(bindings), 4, group)
        table.setHorizontalHeaderLabels(["Key", "Action", "Type", "Category"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setFocusPolicy(_focus_policy_no_focus())
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        for row, binding in enumerate(bindings):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(format_key_label(binding.key)))
            name_item = QtWidgets.QTableWidgetItem(self._display_label(binding))
            name_item.setToolTip(binding.description)
            table.setItem(row, 1, name_item)
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(binding.behavior.title()))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(binding.category))

        table.verticalHeader().setDefaultSectionSize(28)
        table.setMinimumHeight(min(360, 28 * max(4, len(bindings)) + 40))
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        layout.addWidget(table)
        return group

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _build_key_lookup(self, bindings: Sequence) -> None:
        special_map = {
            "space": ["Key_Space"],
            "esc": ["Key_Escape"],
            "escape": ["Key_Escape"],
            "enter": ["Key_Return", "Key_Enter"],
            "return": ["Key_Return", "Key_Enter"],
            "up": ["Key_Up"],
            "down": ["Key_Down"],
            "left": ["Key_Left"],
            "right": ["Key_Right"],
        }
        for binding in bindings:
            key = binding.key.strip().lower()
            if len(key) == 1:
                self._char_action_map[key] = binding.action
                qt_name = f"Key_{key.upper()}"
                try:
                    self._special_action_map[_qt_key(qt_name)] = binding.action
                except AttributeError:
                    pass
            elif key in special_map:
                for qt_name in special_map[key]:
                    self._special_action_map[_qt_key(qt_name)] = binding.action

    def _handle_key_event(self, event, *, is_press: bool) -> bool:
        if event.isAutoRepeat():
            return False

        action = self._action_from_event(event)
        if not action:
            return False

        focus_widget = QtWidgets.QApplication.focusWidget()
        if focus_widget in {self._fsm_input, getattr(self, "_tts_input", None)} and action != "quit":
            return False

        if action in self._controller.hold_actions:
            self._controller.set_hold_state(action, is_press)
            self._update_active_holds_label()
            return True

        if is_press:
            self._controller.trigger_press(action)
            if action == "quit":
                self.close()
            self._update_status_labels()
            return True
        return False

    def _action_from_event(self, event) -> Optional[str]:
        text = event.text()
        if text:
            char = text.lower()
            if len(char) == 1 and char in self._char_action_map:
                return self._char_action_map[char]

        key = event.key()
        return self._special_action_map.get(key)

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _on_action_button(self, action: str) -> None:
        try:
            self._controller.trigger_press(action)
        except KeyError:
            return
        if action == "quit":
            self.close()
            return
        self._update_status_labels()
        self._update_tts_status("")

    def _on_send_tts(self) -> None:
        if not self._tts_interface.available:
            QtWidgets.QMessageBox.warning(
                self,
                "Audio Unavailable",
                self._tts_interface.error_message or "TTS is not available on this system.",
            )
            return

        message = self._tts_input.text().strip()
        if not message:
            return

        try:
            self._tts_interface.speak(message)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid TTS Message", str(exc))
            return
        except Exception as exc:  # pragma: no cover - hardware / SDK dependent
            QtWidgets.QMessageBox.critical(
                self,
                "TTS Failed",
                f"Unable to send text-to-speech command:\n{exc}",
            )
            return

        self._update_tts_status(f'Sent TTS: "{message}"')
        self._tts_input.clear()

    def _on_submit_custom_fsm(self) -> None:
        text = self._fsm_input.text().strip()
        if not text:
            return
        try:
            fsm_id = int(text, 0)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid FSM ID",
                "Please enter a valid integer value (decimal or 0x hex).",
            )
            return

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm FSM Command",
            f"Send FSM ID {fsm_id}? This will latch for {MODE_HOLD_DURATION:.1f}s.",
        )

        yes_button = (
            QtWidgets.QMessageBox.StandardButton.Yes  # type: ignore[attr-defined]
        )
        if confirm != yes_button:
            return
        try:
            self._controller.queue_custom_fsm(
                fsm_id, description=f"FSM {fsm_id}"
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            QtWidgets.QMessageBox.critical(
                self,
                "Command Failed",
                f"Failed to schedule FSM command: {exc}",
            )
            return

        self._fsm_input.clear()
        self._update_status_labels()

    def _on_camera_section_toggled(self, expanded: bool) -> None:
        if not expanded and getattr(self, "_camera_panel", None) is not None:
            self._camera_panel.shutdown()

    def _on_tick(self) -> None:
        self._controller.tick()
        self._update_status_labels()
        self._update_active_holds_label()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._controller.shutdown()
        if hasattr(self, "_joint_panel"):
            self._joint_panel.shutdown()
        if hasattr(self, "_camera_panel"):
            self._camera_panel.shutdown()
        if hasattr(self, "_tts_interface"):
            self._tts_interface.close()
        if hasattr(self, "_state_timer"):
            self._state_timer.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------

    def _update_status_labels(self) -> None:
        state = self._controller.state
        velocity = self._controller.velocity
        fsm = state.fsm_id if state.fsm_id is not None else "—"
        mode = state.mode if state.mode is not None else "—"
        height = self._controller.stand_height
        height_text = "—"
        if height is not None and not math.isnan(height):
            height_text = f"{height:0.3f} m"
        status_text = (
            f"FSM: {fsm}   Mode: {mode}   Height: {height_text}   "
            f"Velocity vx: {velocity[0]:+0.3f}  vy: {velocity[1]:+0.3f}  ω: {velocity[2]:+0.3f}"
        )
        self._status_label.setText(status_text)
        self._event_label.setText(self._controller.last_event)

    def _update_tts_status(self, message: str) -> None:
        if getattr(self, "_tts_status_label", None) is None:
            return
        if hasattr(self, "_tts_interface") and not self._tts_interface.available:
            return
        if message:
            self._tts_status_label.setText(message)
            self._tts_status_label.setStyleSheet("color: #356859;")
        else:
            self._tts_status_label.setText("Voice 1 • Volume 80% (defaults)")
            self._tts_status_label.setStyleSheet("color: #4a5a73;")

    def _update_active_holds_label(self) -> None:
        active = self._controller.active_hold_actions
        if not active:
            self._holds_label.setText("Motion inputs: idle")
            return
        labels = []
        for action in sorted(active):
            binding = self._binding_for_action(action)
            labels.append(
                f"{self._display_label(binding)} ({format_key_label(binding.key)})"
            )
        self._holds_label.setText(f"Motion inputs: {' • '.join(labels)}")

    def _binding_for_action(self, action: str):
        for binding in self._controller.bindings:
            if binding.action == action:
                return binding
        raise KeyError(action)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iface",
        default=DEFAULT_INTERFACE,
        help="Network interface connected to the robot (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    try:
        attr = Qt.ApplicationAttribute.AA_EnableHighDpiScaling  # type: ignore[attr-defined]
        QtCore.QCoreApplication.setAttribute(attr, True)
    except AttributeError:  # PyQt5 fallback
        try:
            QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # type: ignore[attr-defined]
        except AttributeError:
            pass

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ELLISa G1 Control Panel")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)

    window = ControlPanelWindow(interface=args.iface)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - GUI entry point.
    sys.exit(main())
