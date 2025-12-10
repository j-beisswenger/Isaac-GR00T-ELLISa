#!/usr/bin/env python3
"""
Core locomotion control helpers for the Unitree G1.

This module organises all keyboard-style bindings and exposes a reusable
``TeleopController`` class that can be driven by different user interfaces
(command-line, GUI, automated scripts, …).  It does not provide any user
interface on its own; see ``control_panel.py`` for the PyQt control surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from const import (
    DEFAULT_INTERFACE,
    DOUBLE_PRESS_WINDOW,
    FORWARD_SPEED,
    LATERAL_SPEED,
    MODE_HOLD_DURATION,
    MODE_HOLD_INTERVAL,
    SEND_PERIOD,
    STATE_POLL_PERIOD,
    YAW_RATE,
)
from utilities import (
    KeyboardBinding,
    LocoInitResult,
    LocoState,
    get_loco_state,
    get_stand_height,
    initialise_loco_client,
)

try:  # Optional import for typing only.
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ModuleNotFoundError:  # pragma: no cover - we only need the type.
    LocoClient = object  # type: ignore[assignment]


# ============================================================================
# Binding metadata helpers
# ============================================================================


def format_key_label(raw: str) -> str:
    """Return a user-friendly label for ``KeyboardBinding.key``."""
    lookup = {
        "up": "UP",
        "down": "DOWN",
        "left": "LEFT",
        "right": "RIGHT",
        "space": "SPACE",
        "esc": "ESC",
        "enter": "ENTER",
        "return": "ENTER",
    }
    return lookup.get(raw.lower(), raw.upper())


def group_bindings_by_category(
    bindings: Sequence[KeyboardBinding],
) -> Dict[str, Tuple[KeyboardBinding, ...]]:
    """Organise bindings by their ``category`` for UI presentation."""
    grouped: Dict[str, list[KeyboardBinding]] = {}
    for binding in bindings:
        grouped.setdefault(binding.category, []).append(binding)
    return {category: tuple(items) for category, items in grouped.items()}


# ============================================================================
# Default binding set
# ============================================================================


DEFAULT_BINDINGS: Tuple[KeyboardBinding, ...] = (
    KeyboardBinding(
        "forward", "w", "Forward (slow gait)", behavior="hold", category="Motion"
    ),
    KeyboardBinding("backward", "s", "Backward", behavior="hold", category="Motion"),
    KeyboardBinding("left", "a", "Step left", behavior="hold", category="Motion"),
    KeyboardBinding("right", "d", "Step right", behavior="hold", category="Motion"),
    KeyboardBinding("turn_left", "q", "Rotate left", behavior="hold", category="Motion"),
    KeyboardBinding(
        "turn_right", "e", "Rotate right", behavior="hold", category="Motion"
    ),
    KeyboardBinding(
        "stop", "space", "Zero velocities while held", behavior="hold", category="Motion"
    ),
    KeyboardBinding(
        "handshake", "h", "Handshake gesture", behavior="press", category="Gestures"
    ),
    KeyboardBinding(
        "mode_zero_torque",
        "0",
        "Zero torque (FSM 0)",
        behavior="press",
        category="Modes",
    ),
    KeyboardBinding(
        "mode_locked",
        "1",
        "Static balance stand (mode 0)",
        behavior="press",
        category="Modes",
    ),
    KeyboardBinding(
        "mode_sit", "4", "Sit down (FSM 3)", behavior="press", category="Modes"
    ),
    KeyboardBinding(
        "mode_standup",
        "5",
        "Stand-up helper (FSM 4)",
        behavior="press",
        category="Modes",
    ),
    KeyboardBinding(
        "mode_squat", "6", "Squat (FSM 2)", behavior="press", category="Modes"
    ),
    KeyboardBinding(
        "mode_squat_recover",
        "7",
        "Squat → stand (FSM 706)",
        behavior="press",
        category="Recovery",
    ),
    KeyboardBinding(
        "mode_lie_recover",
        "8",
        "Lie → stand (FSM 702)",
        behavior="press",
        category="Recovery",
    ),
    KeyboardBinding(
        "mode_regular_walk",
        "9",
        "Regular walk (FSM 500)",
        behavior="press",
        category="Modes",
    ),
    KeyboardBinding(
        "mode_run", "r", "Run mode (FSM 801)", behavior="press", category="Modes"
    ),
    KeyboardBinding(
        "damp", "x", "Enter Damp (double press)", behavior="press", category="Modes"
    ),
    KeyboardBinding(
        "start",
        "enter",
        "Start balance controller",
        behavior="press",
        category="Utility",
    ),
    KeyboardBinding("quit", "esc", "Exit tele-op", behavior="press", category="System"),
)


# ============================================================================
# Internal helpers
# ============================================================================


@dataclass
class RepeatedCommand:
    """Track scheduled commands that need to be retried for a duration."""

    tag: str
    description: str
    callables: Sequence[Callable[[], None]]
    end_time: float
    interval: float
    next_fire: float = 0.0
    on_finish: Optional[Callable[[], None]] = None


class DoublePressLatch:
    """Helper that detects double presses within a defined time window."""

    def __init__(self, window: float = DOUBLE_PRESS_WINDOW) -> None:
        self._window = window
        self._last_press = 0.0

    def trigger(self) -> bool:
        now = time.time()
        if now - self._last_press <= self._window:
            self._last_press = 0.0
            return True
        self._last_press = now
        return False


# ============================================================================
# Teleop controller
# ============================================================================


class TeleopController:
    """
    High-level locomotion controller driven by keyboard-style bindings.

    The controller performs velocity publishing for hold-style motion controls,
    handles double-press confirmations for critical actions, and exposes a
    consistent API that user interfaces can call into.
    """

    def __init__(
        self,
        *,
        bindings: Sequence[KeyboardBinding] = DEFAULT_BINDINGS,
        interface: str = DEFAULT_INTERFACE,
        init_result: Optional[LocoInitResult] = None,
        send_period: float = SEND_PERIOD,
        state_poll_period: float = STATE_POLL_PERIOD,
    ) -> None:
        if not bindings:
            raise ValueError("TeleopController requires at least one binding.")

        self._bindings: Tuple[KeyboardBinding, ...] = tuple(bindings)
        self._binding_map: Dict[str, KeyboardBinding] = {
            binding.action: binding for binding in self._bindings
        }
        if len(self._binding_map) != len(self._bindings):
            raise ValueError("KeyboardBinding actions must be unique.")

        if init_result is None:
            init_result = initialise_loco_client(interface=interface)

        self._client: LocoClient = init_result.client  # type: ignore[assignment]
        self._state: LocoState = init_result.state

        self._send_period = send_period
        self._state_poll_period = state_poll_period
        self._last_state_poll = 0.0

        self._last_event = "Ready."
        self._last_timestamp = 0.0
        self._last_sent: Tuple[Optional[float], Optional[float], Optional[float]] = (
            None,
            None,
            None,
        )
        self._velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._velocity_dirty = True
        self._stand_height: Optional[float] = None

        self._hold_actions = {b.action for b in self._bindings if b.behavior == "hold"}
        self._press_actions = {b.action for b in self._bindings if b.behavior == "press"}
        self._active_holds: Set[str] = set()
        self._repeated_commands: list[RepeatedCommand] = []

        self._running = True

        self._damp_latch = DoublePressLatch()
        self._zero_torque_latch = DoublePressLatch()
        self._sit_latch = DoublePressLatch()

        self._action_handlers: Dict[str, Callable[[], Optional[bool]]] = {
            "handshake": self._client.ShakeHand,
            "mode_zero_torque": self._handle_mode_zero,
            "mode_locked": self._handle_mode_locked,
            "mode_sit": self._handle_mode_sit,
            "mode_standup": self._handle_mode_standup,
            "mode_squat": self._handle_mode_squat,
            "mode_squat_recover": self._handle_mode_squat_recover,
            "mode_lie_recover": self._handle_mode_lie_recover,
            "mode_regular_walk": self._handle_mode_regular_walk,
            "mode_run": self._handle_mode_run,
            "damp": self._handle_damp,
            "start": self._handle_start,
            "quit": self._handle_quit,
        }
        self._state_actions = {"start"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def bindings(self) -> Tuple[KeyboardBinding, ...]:
        return self._bindings

    @property
    def hold_actions(self) -> Set[str]:
        return set(self._hold_actions)

    @property
    def press_actions(self) -> Set[str]:
        return set(self._press_actions)

    @property
    def active_hold_actions(self) -> Set[str]:
        return set(self._active_holds)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return self._velocity

    @property
    def last_event(self) -> str:
        return self._last_event

    @property
    def state(self) -> LocoState:
        return self._state

    @property
    def stand_height(self) -> Optional[float]:
        return self._stand_height

    @property
    def running(self) -> bool:
        return self._running

    def set_hold_state(self, action: str, active: bool) -> None:
        if action not in self._hold_actions:
            raise KeyError(f"'{action}' is not a hold action.")
        if active:
            if action not in self._active_holds:
                self._active_holds.add(action)
                self._velocity_dirty = True
        else:
            if action in self._active_holds:
                self._active_holds.discard(action)
                self._velocity_dirty = True

    def clear_hold_actions(self) -> None:
        if self._active_holds:
            self._active_holds.clear()
            self._velocity_dirty = True

    def trigger_press(self, action: str) -> None:
        if action not in self._press_actions:
            raise KeyError(f"'{action}' is not a press action.")
        handler = self._action_handlers.get(action)
        if handler is None:
            self._log_event(f"No handler defined for '{action}'.")
            return
        binding = self._binding_map[action]
        try:
            result = handler()
            if result is not False:
                self._log_event(f"{binding.description} command sent.")
        except Exception as exc:  # pragma: no cover - depends on hardware state.
            self._log_event(f"{binding.description} failed: {exc}")
        if action in self._state_actions:
            self.refresh_state()

    def queue_custom_fsm(self, fsm_id: int, *, description: Optional[str] = None) -> None:
        desc = description or f"Custom FSM {fsm_id}"
        self._queue_mode_change(desc, (lambda fid=fsm_id: self._client.SetFsmId(fid),))

    def tick(self) -> None:
        """Advance the controller state; call regularly from the UI loop."""
        if not self._running:
            return

        self._process_repeated_commands()

        now = time.time()
        vx, vy, omega = self._compute_velocity()
        self._velocity = (vx, vy, omega)

        should_send = (
            self._velocity_dirty
            or self._last_sent[0] is None
            or (vx, vy, omega) != self._last_sent
            or (now - self._last_timestamp) >= self._send_period
        )
        if should_send:
            self._send_velocity(vx, vy, omega)
            self._last_sent = (vx, vy, omega)
            self._last_timestamp = now
            self._velocity_dirty = False

        if now - self._last_state_poll >= self._state_poll_period:
            self.refresh_state()

    def refresh_state(self) -> None:
        try:
            self._state = get_loco_state(self._client)
        except Exception:  # pragma: no cover - depends on hardware state.
            pass
        try:
            self._stand_height = get_stand_height(self._client)
        except Exception:  # pragma: no cover - depends on hardware state.
            pass
        self._last_state_poll = time.time()

    def shutdown(self) -> None:
        """Stop active motions and cancel scheduled commands."""
        self._running = False
        self._repeated_commands.clear()
        try:
            self._client.StopMove()
        except Exception:  # pragma: no cover - depends on hardware state.
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_event(self, message: str) -> None:
        self._last_event = message

    def _send_velocity(self, vx: float, vy: float, omega: float) -> None:
        try:
            self._client.Move(vx, vy, omega, continous_move=True)
        except Exception as exc:  # pragma: no cover - depends on hardware state.
            self._log_event(f"Move failed: {exc}")

    def _compute_velocity(self) -> Tuple[float, float, float]:
        vx = vy = omega = 0.0
        holds = self._active_holds

        if "forward" in holds and "backward" not in holds:
            vx = FORWARD_SPEED
        elif "backward" in holds and "forward" not in holds:
            vx = -FORWARD_SPEED

        if "left" in holds and "right" not in holds:
            vy = LATERAL_SPEED
        elif "right" in holds and "left" not in holds:
            vy = -LATERAL_SPEED

        if "turn_left" in holds and "turn_right" not in holds:
            omega = YAW_RATE
        elif "turn_right" in holds and "turn_left" not in holds:
            omega = -YAW_RATE

        if "stop" in holds:
            vx = vy = omega = 0.0

        return (vx, vy, omega)

    def _process_repeated_commands(self) -> None:
        if not self._repeated_commands:
            return

        now = time.time()
        for command in list(self._repeated_commands):
            if now >= command.end_time:
                self._log_event(f"{command.description} command latched.")
                if command.on_finish is not None:
                    try:
                        command.on_finish()
                    except Exception as exc:
                        self._log_event(f"{command.description} follow-up failed: {exc}")
                self._repeated_commands.remove(command)
                continue

            if now >= command.next_fire:
                try:
                    for func in command.callables:
                        func()
                except Exception as exc:
                    self._log_event(f"{command.description} failed: {exc}")
                    self._repeated_commands.remove(command)
                    continue
                command.next_fire = now + command.interval

    def _schedule_repeat(
        self,
        tag: str,
        description: str,
        callables: Sequence[Callable[[], None]],
        *,
        duration: float,
        interval: float,
        on_finish: Optional[Callable[[], None]] = None,
    ) -> None:
        if not callables:
            return

        for existing in list(self._repeated_commands):
            if existing.tag == tag:
                self._repeated_commands.remove(existing)

        command = RepeatedCommand(
            tag=tag,
            description=description,
            callables=tuple(callables),
            end_time=time.time() + duration,
            interval=interval,
            next_fire=0.0,
            on_finish=on_finish,
        )
        self._repeated_commands.append(command)
        self._log_event(f"{description}: holding command for {duration:.1f}s.")

    def _queue_mode_change(
        self,
        description: str,
        callables: Sequence[Callable[[], None]],
        *,
        pre_action: Optional[Callable[[], None]] = None,
    ) -> None:
        if pre_action is not None:
            try:
                pre_action()
            except Exception as exc:
                self._log_event(f"{description} pre-action failed: {exc}")
        self._schedule_repeat(
            "mode",
            description,
            callables,
            duration=MODE_HOLD_DURATION,
            interval=MODE_HOLD_INTERVAL,
            on_finish=self.refresh_state,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_quit(self) -> Optional[bool]:
        self._running = False
        self._log_event("Exit requested.")
        return None

    def _handle_start(self) -> Optional[bool]:
        try:
            self._client.Start()
            self._client.StopMove()
        except Exception as exc:
            self._log_event(f"Failed to start balance controller: {exc}")
            return False
        return None

    def _handle_damp(self) -> Optional[bool]:
        if self._damp_latch.trigger():
            self._queue_mode_change("Damp mode", (lambda: self._client.SetFsmId(1),))
            return None
        self._log_event(
            f"Press X again within {DOUBLE_PRESS_WINDOW:.1f}s to confirm damp mode."
        )
        return False

    def _handle_mode_zero(self) -> Optional[bool]:
        if self._zero_torque_latch.trigger():
            self._queue_mode_change(
                "Zero torque",
                (lambda: self._client.SetFsmId(0),),
                pre_action=self._client.StopMove,
            )
            return None
        self._log_event(
            f"Press 0 again within {DOUBLE_PRESS_WINDOW:.1f}s to confirm zero torque."
        )
        return False

    def _handle_mode_locked(self) -> Optional[bool]:
        self._queue_mode_change(
            "Static balance stand",
            (
                lambda: self._client.SetBalanceMode(0),
                lambda: self._client.SetFsmId(200),
            ),
            pre_action=self._client.StopMove,
        )
        return None

    def _handle_mode_sit(self) -> Optional[bool]:
        if self._sit_latch.trigger():
            self._queue_mode_change("Sit down", (lambda: self._client.SetFsmId(3),))
            return None
        self._log_event(
            f"Press 4 again within {DOUBLE_PRESS_WINDOW:.1f}s to confirm sit down."
        )
        return False

    def _handle_mode_standup(self) -> Optional[bool]:
        self._queue_mode_change("Stand-up helper", (lambda: self._client.SetFsmId(4),))
        return None

    def _handle_mode_squat(self) -> Optional[bool]:
        self._queue_mode_change("Squat", (lambda: self._client.SetFsmId(2),))
        return None

    def _handle_mode_squat_recover(self) -> Optional[bool]:
        self._queue_mode_change("Squat → stand", (lambda: self._client.SetFsmId(706),))
        return None

    def _handle_mode_lie_recover(self) -> Optional[bool]:
        self._queue_mode_change("Lie → stand", (lambda: self._client.SetFsmId(702),))
        return None

    def _handle_mode_regular_walk(self) -> Optional[bool]:
        self._queue_mode_change("Regular walk", (lambda: self._client.SetFsmId(500),))
        return None

    def _handle_mode_run(self) -> Optional[bool]:
        self._queue_mode_change("Run mode", (lambda: self._client.SetFsmId(801),))
        return None


__all__ = [
    "TeleopController",
    "DEFAULT_BINDINGS",
    "format_key_label",
    "group_bindings_by_category",
]


if __name__ == "__main__":  # pragma: no cover - manual usage hint.
    print("keyboard_teleop.py only provides reusable control helpers.")
    print("Launch control_panel.py for the interactive experience.")
