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
    parser = argparse.ArgumentParser(description="Replay recorded LeRobot dataset using pylibfranka.")
    parser.add_argument("--robot-ip", required=True, help="IP address of the Franka robot (e.g., 192.168.1.100).")
    parser.add_argument("--local-dir", default="./lerobot_data", help="Directory where the LeRobot dataset is stored.")
    parser.add_argument("--repo-id", default="local/franka_gello_teleop", help="Dataset repo id.")
    parser.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")
    parser.add_argument("--fps", type=int, default=15, help="FPS at which the data was recorded.")
    return parser.parse_args()


def get_episode_actions(dataset_root: Path, repo_id: str, episode_index: int) -> np.ndarray:
    """
    加载数据集并提取指定 Episode 的 Action 序列。
    假设 Action 的维度为 8 (前7维为关节角度，第8维为夹爪开合度)。
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


def replay_trajectory(robot_ip: str, actions: np.ndarray, fps: int) -> None:
    """
    使用 pylibfranka 回放动作。
    手臂使用 1000Hz 实时插值控制，夹爪在后台线程进行同步控制。
    """
    action_dim = actions.shape[1]
    total_frames = len(actions)
    time_per_frame = 1.0 / fps
    total_duration = total_frames * time_per_frame

    # 记录控制循环的时间，用于同步夹爪
    shared_state = {"time_elapsed": 0.0, "is_running": True}

    def gripper_worker() -> None:
        """
        处理夹爪的后台线程。因为 gripper.move 是阻塞调用，
        不能把它放在 1000Hz 的手臂控制回调函数中。
        """
        if action_dim < 8:
            return
            
        try:
            gripper = pylibfranka.Gripper(robot_ip)
            current_gripper_width = actions[0, 7]
            # 移动到初始夹爪位置 (参数：目标宽度，速度)
            gripper.move(current_gripper_width, 0.1) 
            
            last_frame_idx = 0
            while shared_state["is_running"] and shared_state["time_elapsed"] < total_duration:
                # 根据当前运行时间估算执行到了哪一帧
                current_frame_idx = int(shared_state["time_elapsed"] / time_per_frame)
                
                if current_frame_idx > last_frame_idx and current_frame_idx < total_frames:
                    target_width = actions[current_frame_idx, 7]
                    # 如果夹爪指令有明显变化 (大于 2mm)，则发送动作
                    if abs(target_width - current_gripper_width) > 0.002:
                        gripper.move(target_width, 0.1)
                        current_gripper_width = target_width
                    last_frame_idx = current_frame_idx
                
                time.sleep(0.01) # 降低CPU占用
        except Exception as e:
            print(f"[Gripper Error] {e}")

    # 初始化并启动夹爪线程
    gripper_thread = threading.Thread(target=gripper_worker)
    gripper_thread.start()

    # 连接机器人
    print(f"Connecting to Franka at {robot_ip}...")
    robot = pylibfranka.Robot(robot_ip)
    
    # 可以选择设置初始碰撞行为等...
    # robot.set_collision_behavior(...)

    print("Starting trajectory playback...")
    
    def control_callback(robot_state: pylibfranka.RobotState, time_step: pylibfranka.Duration):
        # 更新已流逝时间
        shared_state["time_elapsed"] += time_step.to_sec()
        t = shared_state["time_elapsed"]
        
        frame_idx = int(t / time_per_frame)
        
        # 如果播放完毕，发送 MotionFinished
        if frame_idx >= total_frames - 1:
            target_q = actions[-1, :7].tolist()
            shared_state["is_running"] = False
            return pylibfranka.MotionFinished(pylibfranka.JointPositions(target_q))
            
        # --- 核心：15Hz 到 1000Hz 的线性插值平滑处理 ---
        alpha = (t % time_per_frame) / time_per_frame
        q_start = actions[frame_idx, :7]
        q_end = actions[frame_idx + 1, :7]
        
        target_q = q_start * (1.0 - alpha) + q_end * alpha
        
        return pylibfranka.JointPositions(target_q.tolist())

    try:
        # robot.control 是一个阻塞调用，会以 1000Hz 的频率循环调用我们提供的 callback
        robot.control(control_callback)
        print("Trajectory playback finished successfully.")
    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        shared_state["is_running"] = False
        gripper_thread.join()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.local_dir).expanduser()
    actions = get_episode_actions(dataset_root, args.repo_id, args.episode)
    replay_trajectory(args.robot_ip, actions, args.fps)

if __name__ == "__main__":
    main()