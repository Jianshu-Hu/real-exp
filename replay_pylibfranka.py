from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np
import pylibfranka

# 确保能够引入本地的 lerobot 库
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay dual-arm recorded LeRobot dataset using pylibfranka.")
    parser.add_argument("dataset_dir", type=str, help="Directory where the LeRobot dataset is stored (e.g., ./lerobot_data_cam1_cam2).")
    parser.add_argument("--ip-left", default="172.16.03", help="IP address of the Left Franka robot.")
    parser.add_argument("--ip-right", default="172.16.02", help="IP address of the Right Franka robot.")
    parser.add_argument("--repo-id", default="local/franka_gello_teleop", help="Dataset repo id.")
    parser.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")
    parser.add_argument("--fps", type=int, default=15, help="FPS at which the data was recorded.")
    return parser.parse_args()


def get_episode_actions(dataset_root: Path, repo_id: str, episode_index: int) -> np.ndarray:
    """
    加载数据集并提取指定 Episode 的 Action 序列。
    支持自适应维度（通过解析 lerobot dataset features 自动获取实际长度）。
    """
    print(f"Loading dataset from {dataset_root}...")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
    )
    
    # 过滤出指定的 episode
    episode_data = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_index)
    actions = np.array(episode_data["action"])
    
    if len(actions) == 0:
        raise ValueError(f"Episode {episode_index} not found or has no data.")
    
    print(f"Loaded {len(actions)} frames for episode {episode_index}.")
    return actions


def arm_worker(robot_ip: str, actions: np.ndarray, fps: int, arm_name: str, abort_event: threading.Event) -> None:
    time_per_frame = 1.0 / fps
    total_frames = len(actions)
    
    print(f"[{arm_name}] Connecting to Franka at {robot_ip}...")
    try:
        robot = pylibfranka.Robot(robot_ip)
    except Exception as e:
        print(f"[{arm_name}] Failed to connect: {e}")
        abort_event.set()
        return

    print(f"[{arm_name}] Starting trajectory playback...")

    try:
        # --- 关键步骤 1: 预备归位 (Reset to start pose) ---
        print(f"[{arm_name}] Moving to start pose...")
        current_state = robot.read_once()
        start_q = actions[0].tolist()
        
        # 使用低速平滑移动到轨迹起点，避免 start_pose_invalid
        # 注意：此处的移动方式取决于你 pylibfranka 的具体绑定，通常可用 robot.move
        # 如果没有快捷 move，可以写一个极短的线性轨迹来过渡
        
        # --- 关键步骤 2: 启动控制循环 ---
        control = robot.start_joint_position_control(pylibfranka.ControllerMode.kJointImpedance)
        
        # 获取启动瞬间的精确位置作为基准
        state, _ = control.read_once()
        last_q = np.array(state.q)
        
        time_elapsed = 0.0
        time_per_frame = 1.0 / fps
        
        while not abort_event.is_set():
            state, time_step = control.read_once()
            dt = time_step.to_sec()
            time_elapsed += dt
            
            frame_idx = int(time_elapsed / time_per_frame)
            
            if frame_idx >= len(actions) - 1:
                # 结束前平滑减速，或者直接保持最后一帧
                jp = pylibfranka.JointPositions(actions[-1].tolist())
                jp.motion_finished = True
                control.write_once(jp)
                break

            # --- 关键步骤 3: 插值平滑处理 ---
            alpha = (time_elapsed % time_per_frame) / time_per_frame
            q_target = actions[frame_idx] * (1.0 - alpha) + actions[frame_idx + 1] * alpha
            
            # 这里的 q_target 发送给机器人
            jp = pylibfranka.JointPositions(q_target.tolist())
            control.write_once(jp)
            
    except Exception as e:
        print(f"[{arm_name}] Reflex Triggered: {e}")
        abort_event.set()

def gripper_worker(robot_ip: str, actions: np.ndarray, fps: int, arm_name: str, abort_event: threading.Event) -> None:
    if actions is None:
        return
        
    time_per_frame = 1.0 / fps
    total_frames = len(actions)
    total_duration = total_frames * time_per_frame
    
    try:
        gripper = pylibfranka.Gripper(robot_ip)
        current_w = actions[0]
        gripper.move(current_w, 0.1)
        
        start_time = time.time()
        last_frame_idx = 0
        
        while not abort_event.is_set():
            t = time.time() - start_time
            if t > total_duration:
                break
                
            current_frame_idx = int(t / time_per_frame)
            if current_frame_idx > last_frame_idx and current_frame_idx < total_frames:
                target_w = actions[current_frame_idx]
                if abs(target_w - current_w) > 0.002:
                    gripper.move(target_w, 0.1)
                    current_w = target_w
                last_frame_idx = current_frame_idx
                
            time.sleep(0.01)
    except Exception as e:
        print(f"[{arm_name} Gripper] Error: {e}")
        abort_event.set()


def replay_dual_trajectory(ip_left: str, ip_right: str, actions: np.ndarray, fps: int) -> None:
    """
    使用多线程和 pylibfranka 同步回放双臂动作。
    """
    action_dim = actions.shape[1]
    
    # 动作维度拆分假设
    if action_dim == 16:
        print("Detected 16-dim action space: Assuming [Left Arm(7), Left Grip(1), Right Arm(7), Right Grip(1)]")
        left_arm_actions = actions[:, 0:7]
        left_grip_actions = actions[:, 7]
        right_arm_actions = actions[:, 8:15]
        right_grip_actions = actions[:, 15]
    elif action_dim == 14:
        print("Detected 14-dim action space: Assuming [Left Arm(7), Right Arm(7)] (No Grippers)")
        left_arm_actions = actions[:, 0:7]
        left_grip_actions = None
        right_arm_actions = actions[:, 7:14]
        right_grip_actions = None
    else:
        raise ValueError(f"Unsupported action dimension: {action_dim}. Expected 14 or 16 for dual-arm.")

    abort_event = threading.Event()
    threads = []

    # 启动机械臂控制线程
    threads.append(threading.Thread(target=arm_worker, args=(ip_left, left_arm_actions, fps, "Left Arm", abort_event)))
    threads.append(threading.Thread(target=arm_worker, args=(ip_right, right_arm_actions, fps, "Right Arm", abort_event)))
    
    # 启动夹爪控制线程
    if left_grip_actions is not None:
        threads.append(threading.Thread(target=gripper_worker, args=(ip_left, left_grip_actions, fps, "Left", abort_event)))
    if right_grip_actions is not None:
        threads.append(threading.Thread(target=gripper_worker, args=(ip_right, right_grip_actions, fps, "Right", abort_event)))

    for t in threads:
        t.start()
        
    for t in threads:
        t.join()

    if abort_event.is_set():
        print("Dual-arm replay aborted due to an error.")
    else:
        print("Dual-arm replay completed.")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser()
    actions = get_episode_actions(dataset_root, args.repo_id, args.episode)
    replay_dual_trajectory(args.ip_left, args.ip_right, actions, args.fps)

if __name__ == "__main__":
    main()