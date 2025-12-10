"""
Shared utilities for the G1 control panel scripts.

This module provides:
- Unitree SDK channel initialization helpers
- Video client factory for camera access
- Locomotion client initialization and state management
- Keyboard input tracking for interactive control
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Set
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)

try:  # Optional dependency; the video client is not always available.
    from unitree_sdk2py.go2.video.video_client import VideoClient
except ImportError:  # pragma: no cover - camera preview is optional.
    VideoClient = None  # type: ignore[assignment]

from const import DEFAULT_INTERFACE, DEFAULT_STAND_HEIGHT, READY_FSM_IDS, READY_MODES


# ============================================================================
# Module State
# ============================================================================

_INITIALISED_INTERFACE: Optional[str] = None


# ============================================================================
# SDK Channel Initialization
# ============================================================================


def ensure_channel_factory(interface: str = DEFAULT_INTERFACE) -> None:
    """Initialise the Unitree channel factory once per process."""
    global _INITIALISED_INTERFACE
    if _INITIALISED_INTERFACE == interface:
        return
    ChannelFactoryInitialize(0, interface)
    _INITIALISED_INTERFACE = interface


def create_video_client(
    interface: str = DEFAULT_INTERFACE, *, timeout: float = 3.0
) -> VideoClient:
    """
    Create and initialize a video client for camera access.

    Args:
        interface: Network interface name.
        timeout: Client timeout in seconds.

    Returns:
        Initialized VideoClient instance.

    Raises:
        RuntimeError: If VideoClient is not available.
    """
    if VideoClient is None:  # pragma: no cover - depends on SDK install.
        raise RuntimeError(
            "unitree_sdk2py.go2.video.video_client is not available on this system."
        )
    ensure_channel_factory(interface)
    client = VideoClient()
    client.SetTimeout(timeout)
    client.Init()
    return client


# ============================================================================
# Locomotion Client Helpers
# ============================================================================


@dataclass(frozen=True)
class KeyboardBinding:
    """Describe a keyboard shortcut supported by the tele-op helpers."""

    action: str
    key: str
    description: str
    behavior: str = "press"  # "press", "hold", or "pulse"
    category: str = "General"

    def __post_init__(self) -> None:
        if self.behavior not in ("press", "hold", "pulse"):
            raise ValueError(f"Unknown binding behavior '{self.behavior}'.")


@dataclass(frozen=True)
class LocoState:
    """Snapshot of the robot's high-level locomotion state."""

    fsm_id: Optional[int]
    mode: Optional[int]


@dataclass(frozen=True)
class LocoInitResult:
    """Return value for initialise_loco_client."""

    client: LocoClient
    state: LocoState
    bringup_performed: bool


class KeyPressTracker:
    """Track pressed keys using pynput so we can react to key-up events."""

    def __init__(self, bindings: Sequence[KeyboardBinding]):
        if not bindings:
            raise ValueError("KeyPressTracker requires at least one binding.")

        try:
            from pynput.keyboard import Key, KeyCode, Listener  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Keyboard tracking requires the 'pynput' package. "
                "Install it with 'pip install pynput'."
            ) from exc

        self._Key = Key
        self._KeyCode = KeyCode
        self._Listener = Listener
        self._bindings = list(bindings)
        self._pressed: Set[object] = set()
        self._lock = threading.Lock()
        self._binding_by_action = {
            binding.action: binding for binding in self._bindings
        }
        if len(self._binding_by_action) != len(self._bindings):
            raise ValueError("KeyboardBinding actions must be unique.")
        self._key_map = {
            binding.action: self._resolve_key(binding.key) for binding in self._bindings
        }
        self._listener = self._Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._active:
            return
        self._listener.start()
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self._listener.stop()
        try:
            self._listener.join(1.0)
        except RuntimeError:
            pass
        self._active = False

    def __enter__(self) -> "KeyPressTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # State query helpers
    # ------------------------------------------------------------------

    @property
    def bindings(self) -> Sequence[KeyboardBinding]:
        return tuple(self._bindings)

    def is_active(self, action: str) -> bool:
        if action not in self._key_map:
            raise KeyError(f"Unknown action '{action}'.")
        target = self._key_map[action]
        with self._lock:
            return target in self._pressed

    def active_actions(self) -> Sequence[str]:
        active: list[str] = []
        with self._lock:
            for action, key_obj in self._key_map.items():
                if key_obj in self._pressed:
                    active.append(action)
        return tuple(active)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key_spec: str) -> object:
        name = key_spec.strip().lower()
        lookup = {
            "up": self._Key.up,
            "arrow_up": self._Key.up,
            "down": self._Key.down,
            "arrow_down": self._Key.down,
            "left": self._Key.left,
            "arrow_left": self._Key.left,
            "right": self._Key.right,
            "arrow_right": self._Key.right,
            "space": self._Key.space,
            "esc": self._Key.esc,
            "escape": self._Key.esc,
            "enter": self._Key.enter,
            "return": self._Key.enter,
        }

        if len(name) == 1:
            return name
        if name in lookup:
            return lookup[name]
        raise ValueError(f"Unsupported key specifier '{key_spec}'.")

    def _on_press(self, key: object) -> None:
        with self._lock:
            if isinstance(key, self._KeyCode) and key.char is not None:
                self._pressed.add(key.char.lower())
            else:
                self._pressed.add(key)

    def _on_release(self, key: object) -> None:
        with self._lock:
            if isinstance(key, self._KeyCode) and key.char is not None:
                self._pressed.discard(key.char.lower())
            else:
                self._pressed.discard(key)


def _rpc_get_value(client: LocoClient, api_id: int) -> Optional[object]:
    """Helper to call RPC and extract the 'data' payload as-is."""
    try:
        code, data = client._Call(api_id, "{}")  # type: ignore[attr-defined]
        if code == 0 and data:
            return json.loads(data).get("data")
    except Exception:
        pass
    return None


def _rpc_get_int(client: LocoClient, api_id: int) -> Optional[int]:
    """Helper to call RPC and convert result to int when possible."""
    value = _rpc_get_value(client, api_id)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rpc_get_float(client: LocoClient, api_id: int) -> Optional[float]:
    """Helper to call RPC and convert result to float when possible."""
    value = _rpc_get_value(client, api_id)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_loco_state(client: LocoClient) -> LocoState:
    """Query the current locomotion FSM state."""
    return LocoState(
        fsm_id=_rpc_get_int(client, ROBOT_API_ID_LOCO_GET_FSM_ID),
        mode=_rpc_get_int(client, ROBOT_API_ID_LOCO_GET_FSM_MODE),
    )


def get_stand_height(client: LocoClient) -> Optional[float]:
    """Return the current stand height if available."""
    from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_STAND_HEIGHT

    return _rpc_get_float(client, ROBOT_API_ID_LOCO_GET_STAND_HEIGHT)


def _is_ready_state(
    state: LocoState,
    ready_fsm_ids: Sequence[int],
    ready_modes: Optional[Sequence[int]],
) -> bool:
    """Check if the robot is in a ready state."""
    if state.fsm_id is None:
        return False
    if state.fsm_id not in ready_fsm_ids:
        return False
    if not ready_modes:
        return True
    if state.mode is None:
        return True
    return state.mode in ready_modes


def initialise_loco_client(
    interface: str = DEFAULT_INTERFACE,
    *,
    stand_height: float = DEFAULT_STAND_HEIGHT,
    balance_mode: Optional[int] = 0,
    settle_delay: float = 0.6,
    timeout: float = 10.0,
    ready_fsm_ids: Sequence[int] = READY_FSM_IDS,
    ready_modes: Sequence[int] = READY_MODES,
    bringup_if_needed: bool = False,
    zero_velocity_on_attach: bool = True,
) -> LocoInitResult:
    """
    Initialise a G1 loco client while respecting the current gait setup.

    By default the function simply attaches to the live locomotion service so
    existing "regular" or "running" modes are preserved.  If the robot is not
    yet standing and *bringup_if_needed* is set to ``True`` the helper performs
    the usual high-level bring-up sequence (damp → stand-up → balance/start).
    """
    ensure_channel_factory(interface)

    client = LocoClient()
    client.SetTimeout(timeout)
    client.Init()

    state = get_loco_state(client)
    ready = _is_ready_state(state, ready_fsm_ids, ready_modes)
    bringup = bringup_if_needed and not ready

    if not bringup:
        if zero_velocity_on_attach:
            try:
                client.StopMove()
            except Exception:
                pass
        return LocoInitResult(
            client=client, state=get_loco_state(client), bringup_performed=False
        )

    client.Damp()
    if settle_delay > 0.0:
        time.sleep(settle_delay)

    client.SetFsmId(4)  # stand-up behaviour
    if settle_delay > 0.0:
        time.sleep(max(settle_delay, 1.0))

    if stand_height is not None:
        client.SetStandHeight(max(0.0, float(stand_height)))
        if settle_delay > 0.0:
            time.sleep(settle_delay)

    if balance_mode is not None:
        client.BalanceStand(balance_mode)
        if settle_delay > 0.0:
            time.sleep(settle_delay)

    client.Start()
    if zero_velocity_on_attach:
        client.StopMove()

    return LocoInitResult(
        client=client, state=get_loco_state(client), bringup_performed=True
    )


__all__ = [
    "ensure_channel_factory",
    "create_video_client",
    "KeyboardBinding",
    "LocoState",
    "LocoInitResult",
    "KeyPressTracker",
    "get_loco_state",
    "get_stand_height",
    "initialise_loco_client",
]
