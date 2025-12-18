"""
Minimal GR00T → Unitree G1 upper-body client (7-DoF arms, waist locked to 0).

This variant removes debug-only helpers and keeps a concise control loop:
- Streams current arm state and the front RGB camera to GR00T.
- Sends returned joint trajectories to the robot via arm_sdk.
"""

from __future__ import annotations

import argparse
import sys
import time
from sqlite3.dbapi2 import Time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from ELLISA.utilities import create_video_client
from gr00t.eval.service import ExternalRobotInferenceClient


class G1JointIndex:
    """Joint indices for the 29-DoF G1 (including 7-DoF arms)."""

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    kNotUsedJoint = 29


class UpperBodyController:
    """Minimal wrapper around the arm_sdk channel for arms + waist."""

    def __init__(self, interface: str, kp: float = 20.0, kd: float = 1.5) -> None:
        self._crc = CRC()
        self._kp = kp
        self._kd = kd
        self._low_state: LowState_ | None = None
        self._initialized = False

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
        if left_arm.shape != (7,) or right_arm.shape != (7,):
            raise ValueError("Arm targets must be shape (7,).")

        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1  # Enable arm_sdk

        waist_ids = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]
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
    """Return resized RGB frames; raise immediately if unavailable."""

    def __init__(self, interface: str, target_size_x: int = 224, target_size_y: int = 224) -> None:
        self._target_size_x = int(target_size_x)
        self._target_size_y = int(target_size_y)
        self._client = self._init_client(interface)
        self._cv2 = self._init_cv2()
        if not self.ready:
            raise RuntimeError("Camera feed unavailable; ensure video client and OpenCV are installed.")

    @property
    def ready(self) -> bool:
        return self._client is not None and self._cv2 is not None

    def _init_client(self, interface: str) -> Any:
        return create_video_client(interface)

    def _init_cv2(self) -> Any:
        import cv2  # type: ignore

        return cv2

    def next_frame(self) -> np.ndarray:
        if not self.ready or self._client is None or self._cv2 is None:
            raise RuntimeError("Camera feed not ready.")

        code, data = self._client.GetImageSample()
        if code != 0 or not data:
            raise RuntimeError(f"Camera returned error code {code} or empty data.")

        frame_bytes = np.frombuffer(bytes(data), dtype=np.uint8)
        frame_rgb = self._cv2.imdecode(frame_bytes, self._cv2.IMREAD_COLOR)
        if frame_rgb is None:
            raise RuntimeError("Failed to decode camera frame.")
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[0] != self._target_size_y or frame_rgb.shape[1] != self._target_size_x:
            frame_rgb = self._cv2.resize(
                frame_rgb,
                (self._target_size_x, self._target_size_y),
                interpolation=self._cv2.INTER_AREA,
            )
            # save frame
            cv2.imwrite("tmp_img.png", frame_rgb)
        return frame_rgb.astype(np.uint8, copy=False)


def build_observation(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    camera_frame: np.ndarray,
    *,
    hand_fill: float = 0.0,
    task_description: str = "Stack the two boxes.",
) -> Dict[str, Any]:
    if camera_frame is None:
        raise ValueError("Camera frame is required for observation construction.")

    ego_view = camera_frame.reshape(1, *camera_frame.shape)  # * 0  # Making the image black
    hand_stub = np.full((1, 7), hand_fill, dtype=np.float64)

    return {
        "state.left_arm": left_arm.reshape(1, 7).astype(np.float64),
        "state.right_arm": right_arm.reshape(1, 7).astype(np.float64),
        "state.left_hand": hand_stub,
        "state.right_hand": hand_stub,
        "video.cam_right_high": ego_view,
        "annotation.human.action.task_description": [task_description],
    }


def parse_actions(raw_action: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    try:
        left = np.asarray(raw_action["action.left_hand"], dtype=np.float64)
        right = np.asarray(raw_action["action.right_hand"], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing expected action key: {exc}") from exc

    left = np.atleast_2d(left)
    right = np.atleast_2d(right)

    if left.shape[1] != 7 or right.shape[1] != 7:
        raise ValueError(f"Arm actions must have shape (T, 7). Got left {left.shape}, right {right.shape}.")

    horizon = min(left.shape[0], right.shape[0])
    return left[:horizon], right[:horizon]


def interpolate_rollouts(
    prev_left_last: np.ndarray,
    prev_right_last: np.ndarray,
    next_left_first: np.ndarray,
    next_right_first: np.ndarray,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear blend from previous rollout endpoint into next rollout start."""
    blend_steps = max(1, int(steps))
    left_blend = np.linspace(prev_left_last, next_left_first, blend_steps + 1, endpoint=True)[1:]
    right_blend = np.linspace(prev_right_last, next_right_first, blend_steps + 1, endpoint=True)[1:]
    return left_blend, right_blend


def smooth_to_first_action(
    controller: UpperBodyController,
    current_left: np.ndarray,
    current_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    dt: float,
    duration: float = 1.0,
) -> None:
    if duration <= 0.0:
        controller.send_targets(target_left, target_right)
        return

    steps = max(int(duration / dt), 1)
    for idx in range(1, steps + 1):
        ratio = idx / steps
        left_blend = (1.0 - ratio) * current_left + ratio * target_left
        right_blend = (1.0 - ratio) * current_right + ratio * target_right
        controller.send_targets(left_blend, right_blend)
        time.sleep(dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T → G1 upper-body client (7-DoF arms)")
    parser.add_argument("--interface", default="enx00e04c3e5a60", help="Network interface for ChannelFactory")
    parser.add_argument("--host", default="localhost", help="GR00T inference server host")
    parser.add_argument("--port", type=int, default=5555, help="GR00T inference server port")
    parser.add_argument("--dt", type=float, default=0.05, help="Control period (s)")
    parser.add_argument("--kp", type=float, default=40.0, help="Position gain for upper body joints")
    parser.add_argument("--kd", type=float, default=1.5, help="Damping gain for upper body joints")
    parser.add_argument("--smooth_duration", type=float, default=1.0, help="Blend time into first action step")
    parser.add_argument("--hand_fill", type=float, default=0.0, help="Fill value for left/right hand states")
    parser.add_argument(
        "--task_description", default="Stack the two boxes.", help="Instruction string sent to the model"
    )
    parser.add_argument(
        "--interp_steps",
        type=int,
        default=0,
        help="Optional blend steps between consecutive rollouts (0 disables).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ChannelFactoryInitialize(0, args.interface)

    controller = UpperBodyController(args.interface, kp=args.kp, kd=args.kd)
    policy = ExternalRobotInferenceClient(host=args.host, port=args.port)
    camera_feed = CameraFeed(args.interface)

    while not controller.ready:
        time.sleep(0.05)

    smoothed_into_first_action = False
    prev_left_seq: np.ndarray | None = None
    prev_right_seq: np.ndarray | None = None

    try:
        while True:
            # time.sleep(1.0)
            left_q, right_q = controller.current_arm_state()
            frame = camera_feed.next_frame()
            obs = build_observation(
                left_q,
                right_q,
                frame,
                hand_fill=args.hand_fill,
                task_description=args.task_description,
            )
            np.save(f"tmp_obs/{time.time()}.npy", obs)

            action = policy.get_action(obs)
            left_sequence, right_sequence = parse_actions(action)
            horizon = left_sequence.shape[0]
            if horizon == 0:
                continue

            if not np.all(np.isfinite(left_sequence)) or not np.all(np.isfinite(right_sequence)):
                continue

            if (
                args.interp_steps > 0
                and prev_left_seq is not None
                and prev_right_seq is not None
                and prev_left_seq.size > 0
                and prev_right_seq.size > 0
            ):
                blend_left, blend_right = interpolate_rollouts(
                    prev_left_seq[-1],
                    prev_right_seq[-1],
                    left_sequence[0],
                    right_sequence[0],
                    args.interp_steps,
                )
                left_sequence = np.vstack([blend_left, left_sequence])
                right_sequence = np.vstack([blend_right, right_sequence])
                horizon = left_sequence.shape[0]

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
                controller.send_targets(left_sequence[step_idx], right_sequence[step_idx])
                time.sleep(args.dt)

            prev_left_seq = left_sequence
            prev_right_seq = right_sequence
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
