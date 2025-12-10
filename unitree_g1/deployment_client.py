from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any
import numpy as np
import time


def get_robot_state():
    state_left_arm = np.zeros((1, 7), dtype=np.float64)
    state_right_arm = np.zeros((1, 7), dtype=np.float64)
    state_left_hand = np.zeros((1, 7), dtype=np.float64)
    state_right_hand = np.zeros((1, 7), dtype=np.float64)

    video_cam_right_high = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)

    # TODO

    return state_right_hand, state_left_hand, state_right_arm, state_left_arm, video_cam_right_high


def execute_action(raw_action_chunk):
    # raw_action_chunk = {
    #     'action.left_arm': np.zeros((1, 7), dtype=np.float64),
    #     'action.right_arm': np.zeros((1, 7), dtype=np.float64),
    #     'action.left_hand': np.zeros((1, 7), dtype=np.float64),
    #     'action.right_hand': np.zeros((1, 7), dtype=np.float64),
    # }

    # "kLeftShoulderPitch","kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    # "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    # "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2", "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    # "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2", "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"

    # TODO

    breakpoint()


def main():
    policy = ExternalRobotInferenceClient(host="localhost", port=5555)

    while True:
        annotation_human_task_description = ["stack three block"]

        state_right_hand, state_left_hand, state_right_arm, state_left_arm, video_cam_right_high = get_robot_state()

        raw_obs_dict: Dict[str, Any] = {
            "state.right_hand": state_right_hand,
            "state.left_hand": state_left_hand,
            "state.right_arm": state_right_arm,
            "state.left_arm": state_left_arm,
            "video.cam_right_high": video_cam_right_high,
            "annotation.human.task_description": annotation_human_task_description,
        }

        raw_action_chunk = policy.get_action(raw_obs_dict)
        execute_action(raw_action_chunk)

        time.sleep(16.0 / 30.0)  # dataset is recorded with 30fps, and action horizon is 16 frames


if __name__ == "__main__":
    main()
