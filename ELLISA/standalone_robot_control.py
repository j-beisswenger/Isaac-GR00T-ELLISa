#!/usr/bin/env python3
import json
import time
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)


def get_state(client):
    try:
        fsm_code, fsm_data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
        mode_code, mode_data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_MODE, "{}")
        fsm_id = (
            json.loads(fsm_data).get("data") if fsm_code == 0 and fsm_data else None
        )
        mode = (
            json.loads(mode_data).get("data") if mode_code == 0 and mode_data else None
        )
        return fsm_id, mode
    except Exception:
        return None, None


def status_monitor(client, stop_event):
    while not stop_event.is_set():
        fsm_id, mode = get_state(client)
        print(f"\rFSM-ID: {fsm_id:<5} | Mode: {mode:<3} | > ", end="", flush=True)
        time.sleep(0.5)


def main():
    # Initialize connection
    ChannelFactoryInitialize(0, "enp3s0f3u1u4")
    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()
    client.StopMove()

    print("Robot connected. Commands: damp, zero, situp, sit, status, quit")
    print("Continuous status monitoring enabled...\n")

    # Start status monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=status_monitor, args=(client, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()

    while True:
        try:
            cmd = input().strip().lower()

            if cmd in ["quit", "q"]:
                break
            elif cmd == "damp":
                client.SetFsmId(1)
                print("\nDamp mode")
            elif cmd == "zero":
                client.SetFsmId(0)
                print("\nZero torque")
            elif cmd == "situp":
                client.SetFsmId(4)
                client.Start()
                print("\nStand up")
            elif cmd == "sit":
                client.SetFsmId(3)
                print("\nSit down")
            elif cmd == "status":
                fsm_id, mode = get_state(client)
                print(f"\nFSM: {fsm_id}, Mode: {mode}")
            elif cmd:
                print("\nUnknown command")

        except (KeyboardInterrupt, EOFError):
            break

    stop_event.set()
    client.StopMove()
    print("\nStopped.")


if __name__ == "__main__":
    main()

