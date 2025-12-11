"""
GR00T → Unitree G1 upper-body client (7-DoF arms, waist locked to 0).

Notes
-----
- Only the upper body is commanded. No signals are sent to the legs.
- Hand outputs from GR00T are ignored (robot has no hands).
- Waist is held at 0 for all 3 DOF.
- The script expects a running GR00T inference server (ZMQ) reachable at
  --host/--port. It sends the latest arm state and the front RGB camera feed
  (or blank frames if the camera is unavailable) to satisfy the model input
  schema.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from ELLISA.utilities import create_video_client
from gr00t.eval.service import ExternalRobotInferenceClient


class G1JointIndex:
    """Joint indices for the 29-DoF G1 (including 7-DoF arms)."""

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    # Waist (locked to 0)
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    kNotUsedJoint = 29  # SDK flag slot


class UpperBodyController:
    """Wrapper around the arm_sdk channel that commands arms + waist."""

    def __init__(self, interface: str, kp: float = 20.0, kd: float = 1.5) -> None:
        self._crc = CRC()
        self._kp = kp
        self._kd = kd
        self._low_state: LowState_ | None = None
        self._initialized = False

        # Enable the arm_sdk controller.
        # self._switcher = MotionSwitcherClient(interface)
        # self._switcher.Init()
        # # Mode 1 is the upper-body controller (arm_sdk).
        # self._switcher.SwitchMotion(1)

        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._low_state_cb, 10)

    @property
    def ready(self) -> bool:
        return self._initialized and self._low_state is not None

    def _low_state_cb(self, msg: LowState_) -> None:
        self._low_state = msg
        self._initialized = True

    def current_arm_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (left, right) arm joint positions."""
        if self._low_state is None:
            raise RuntimeError("No low state yet.")
        ls = self._low_state
        left_ids = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
        ]
        right_ids = [
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
        ]
        left = np.array([ls.motor_state[i].q for i in left_ids], dtype=np.float64)
        right = np.array([ls.motor_state[i].q for i in right_ids], dtype=np.float64)
        return left, right

    def send_targets(self, left_arm: np.ndarray, right_arm: np.ndarray) -> None:
        """Send 7-DoF arm targets and hold the waist at 0."""

        if left_arm.shape != (7,) or right_arm.shape != (7,):
            raise ValueError("Arm targets must be shape (7,).")

        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1  # Enable arm_sdk

        waist_ids = [
            G1JointIndex.WaistYaw,
            G1JointIndex.WaistRoll,
            G1JointIndex.WaistPitch,
        ]
        for idx in waist_ids:
            motor = cmd.motor_cmd[idx]
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self._kp
            motor.kd = self._kd

        left_ids = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
        ]
        right_ids = [
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
        ]

        for target, idx in zip(left_arm, left_ids):
            motor = cmd.motor_cmd[idx]
            motor.q = float(target)
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self._kp
            motor.kd = self._kd

        for target, idx in zip(right_arm, right_ids):
            motor = cmd.motor_cmd[idx]
            motor.q = float(target)
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self._kp
            motor.kd = self._kd

        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)


class CameraFeed:
    """Thin wrapper around Unitree's VideoClient to return resized RGB frames."""

    def __init__(self, interface: str, target_size_x: int = 640, target_size_y: int = 480) -> None:
        self._target_size_x = int(target_size_x)
        self._target_size_y = int(target_size_y)
        self._client = self._init_client(interface)
        self._cv2 = self._init_cv2()
        self._failures = 0

    @property
    def ready(self) -> bool:
        return self._client is not None and self._cv2 is not None

    def _init_client(self, interface: str) -> Optional[Any]:
        try:
            return create_video_client(interface)
        except Exception as exc:
            print(f"Camera feed unavailable; falling back to blank frames ({exc}).")
            return None

    def _init_cv2(self) -> Optional[Any]:
        try:
            import cv2  # type: ignore

            return cv2
        except Exception as exc:
            print(f"OpenCV not available for camera decode ({exc}); using blank frames.")
            return None

    def next_frame(self) -> Optional[np.ndarray]:
        """Return a (H, W, 3) RGB frame resized to target_size, or None on failure."""
        if not self.ready or self._client is None or self._cv2 is None:
            return None

        try:
            code, data = self._client.GetImageSample()
        except Exception as exc:
            if self._failures < 5:
                print(f"Camera capture error: {exc}")
            self._failures += 1
            return None

        if code != 0 or not data:
            self._failures += 1
            return None

        frame_bytes = np.frombuffer(bytes(data), dtype=np.uint8)
        frame_bgr = self._cv2.imdecode(frame_bytes, self._cv2.IMREAD_COLOR)
        if frame_bgr is None:
            self._failures += 1
            return None

        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[0] != self._target_size_y or frame_rgb.shape[1] != self._target_size_x:
            frame_rgb = self._cv2.resize(
                frame_rgb,
                (self._target_size_x, self._target_size_y),  # width, height
                interpolation=self._cv2.INTER_AREA,
            )

        self._failures = 0
        return frame_rgb.astype(np.uint8, copy=False)


def build_observation(
    left_arm: np.ndarray, right_arm: np.ndarray, camera_frame: Optional[np.ndarray]
) -> Dict[str, Any]:
    """Construct an observation dict for the GR00T server."""

    if camera_frame is None:
        camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ego_view = camera_frame.reshape(1, *camera_frame.shape)

    obs = {
        "state.left_arm": left_arm.reshape(1, 7).astype(np.float64),
        "state.right_arm": right_arm.reshape(1, 7).astype(np.float64),
        "state.left_hand": np.zeros((1, 7), dtype=np.float64),
        "state.right_hand": np.zeros((1, 7), dtype=np.float64),
        # "state.waist": np.zeros((1, 3), dtype=np.float64),
        "video.cam_right_high": ego_view,
        "annotation.human.action.task_description": ["Stack the two boxes."],
    }
    return obs


def parse_actions(raw_action: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the full left/right arm action sequences as float arrays."""

    try:
        left = np.asarray(raw_action["action.left_arm"], dtype=np.float64)
        right = np.asarray(raw_action["action.right_arm"], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing expected action key: {exc}") from exc

    left = np.atleast_2d(left)
    right = np.atleast_2d(right)

    if left.shape[1] != 7 or right.shape[1] != 7:
        raise ValueError(f"Arm actions must have shape (T, 7). Got left {left.shape}, right {right.shape}.")

    horizon = min(left.shape[0], right.shape[0])
    return left[:horizon], right[:horizon]


def smooth_to_first_action(
    controller: UpperBodyController,
    current_left: np.ndarray,
    current_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    dt: float,
    duration: float = 1.0,
) -> None:
    """
    Linearly interpolate from the current pose into the first action step.
    Mirrors the gentle warmup used in high_level examples to avoid jerks.
    """
    if duration <= 0.0:
        controller.send_targets(target_left, target_right)
        return

    steps = max(int(duration / dt), 1)
    for idx in range(1, steps + 1):
        ratio = idx / steps
        left_blend = (1.0 - ratio) * current_left + ratio * target_left
        right_blend = (1.0 - ratio) * current_right + ratio * target_right
        print(
            f"[Smooth] Step {idx}/{steps} | "
            f"left: {np.array2string(left_blend, precision=3)} | "
            f"right: {np.array2string(right_blend, precision=3)}"
        )
        controller.send_targets(left_blend, right_blend)
        time.sleep(dt)


def save_debug_frame(frame_rgb: np.ndarray, save_dir: str = "tmp_frames") -> None:
    """Persist the RGB frame the model saw for later debugging."""
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, frame_bgr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T → G1 upper-body client (7-DoF arms)")
    parser.add_argument("--interface", default="enx00e04c3e5a60", help="Network interface for ChannelFactory")
    parser.add_argument("--host", default="localhost", help="GR00T inference server host")
    parser.add_argument("--port", type=int, default=5555, help="GR00T inference server port")
    parser.add_argument("--dt", type=float, default=0.05, help="Control period (s)")
    parser.add_argument("--kp", type=float, default=20.0, help="Position gain for upper body joints")
    parser.add_argument("--kd", type=float, default=1.5, help="Damping gain for upper body joints")
    parser.add_argument(
        "--smooth_duration",
        type=float,
        default=1.0,
        help="Seconds to blend from the current pose into the first action step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Initializing Unitree SDK channels…")
    ChannelFactoryInitialize(0, args.interface)

    controller = UpperBodyController(args.interface, kp=args.kp, kd=args.kd)
    policy = ExternalRobotInferenceClient(host=args.host, port=args.port)
    camera_feed = CameraFeed(args.interface)
    if camera_feed.ready:
        print("Camera feed ready; streaming live frames to GR00T.")
    else:
        print("Camera feed disabled; sending blank frames.")

    print("Waiting for first lowstate…")
    while not controller.ready:
        time.sleep(0.05)

    last_frame_save_time = 0.0
    smoothed_into_first_action = False
    print("Starting GR00T control loop (Ctrl+C to exit).")
    try:
        while True:
            left_q, right_q = controller.current_arm_state()
            frame = camera_feed.next_frame()
            obs_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            obs = build_observation(left_q, right_q, obs_frame)
            action = policy.get_action(obs)
            left_sequence, right_sequence = parse_actions(action)
            horizon = left_sequence.shape[0]

            if horizon == 0:
                print("Received empty action sequence; skipping this cycle.")
                continue

            if not smoothed_into_first_action:
                smooth_to_first_action(
                    controller,
                    left_q,
                    right_q,
                    left_sequence[0],
                    right_sequence[0],
                    dt=args.dt,
                    duration=args.smooth_duration,
                )
                smoothed_into_first_action = True

            for step_idx in range(horizon):
                left_target = left_sequence[step_idx]
                right_target = right_sequence[step_idx]
                print(
                    f"[GR00T] Step {step_idx + 1}/{horizon} | "
                    f"left: {np.array2string(left_target, precision=3)} | "
                    f"right: {np.array2string(right_target, precision=3)}"
                )
                controller.send_targets(left_target, right_target)

                now = time.time()
                if now - last_frame_save_time >= 1.0:
                    save_debug_frame(obs_frame)
                    last_frame_save_time = now

                time.sleep(args.dt)
    except KeyboardInterrupt:
        print("Shutting down.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
