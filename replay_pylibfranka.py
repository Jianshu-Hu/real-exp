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
        # 建议：在开始前设置较低的碰撞阈值，保护硬件
        robot.set_collision_behavior(
            [20.0]*7, [20.0]*7, [10.0]*7, [10.0]*7,
            [20.0]*6, [20.0]*6, [10.0]*6, [10.0]*6
        )
    except Exception as e:
        print(f"[{arm_name}] Connection failed: {e}")
        abort_event.set()
        return

    try:
        # --- 步骤 1: 预备归位 (Reset to start pose) ---
        # 解决 start_pose_invalid 的核心：先慢速移动到轨迹起点
        print(f"[{arm_name}] Moving slowly to initial trajectory pose...")
        start_q = actions[0].tolist()
        
        # 使用 libfranka 自带的简单运动生成器（通常是 0.1-0.2 的速度系数）
        # 注意：这会自动处理加速度限制，直到到达目标位置才返回
        try:
            # 不同的绑定版本可能叫 robot.move 或 robot.move_to
            robot.move(pylibfranka.JointPositions(start_q), 0.1) 
        except AttributeError:
            # 如果没有 move 方法，请确保手动将机械臂拖到接近起始位的地方
            print(f"[{arm_name}] WARNING: robot.move not found. Ensure arm is at start pose manually.")

        # --- 步骤 2: 准备实时控制 ---
        print(f"[{arm_name}] Start pose reached. Initializing 1000Hz control...")
        
        try:
            mode = pylibfranka.ControllerMode.kJointImpedance
        except AttributeError:
            mode = pylibfranka.ControllerMode.JointImpedance
            
        control = robot.start_joint_position_control(mode)
        
        # 获取启动瞬间的实际状态
        initial_state, _ = control.read_once()
        current_actual_q = np.array(initial_state.q)
        
        time_elapsed = 0.0
        
        # --- 步骤 3: 高频控制循环 ---
        while not abort_event.is_set():
            # 读取当前状态（必须在 1ms 内完成，否则报超时）
            state, time_step = control.read_once()
            
            # 兼容 Duration 对象的不同方法名
            if hasattr(time_step, "to_sec"):
                dt = time_step.to_sec()
            elif hasattr(time_step, "toSec"):
                dt = time_step.toSec()
            else:
                dt = float(time_step)
                
            time_elapsed += dt
            frame_idx = int(time_elapsed / time_per_frame)
            
            # 检查是否到达动作终点
            if frame_idx >= total_frames - 1:
                target_q = actions[-1].tolist()
                jp = pylibfranka.JointPositions(target_q)
                jp.motion_finished = True # 告诉机器人控制器这是最后一帧
                control.write_once(jp)
                break
                
            # 插值计算
            # alpha 是当前时刻在两帧之间的时间比例 [0, 1)
            alpha = (time_elapsed % time_per_frame) / time_per_frame
            q_start = actions[frame_idx]
            q_end = actions[frame_idx + 1]
            
            # 线性插值：解决 15Hz 到 1000Hz 的阶跃问题
            target_q_array = q_start * (1.0 - alpha) + q_end * alpha
            
            # 安全检查：如果插值后的目标与当前实际位置偏差过大，主动中断
            # 这是为了防止数据中的某些异常跳变损坏减速器
            diff = np.linalg.norm(target_q_array - np.array(state.q))
            if diff > 0.05: # 阈值约 3 度
                print(f"[{arm_name}] SAFETY ABORT: Target too far from actual ({diff:.4f} rad)")
                abort_event.set()
                break

            # 下发指令
            jp = pylibfranka.JointPositions(target_q_array.tolist())
            control.write_once(jp)
            
        print(f"[{arm_name}] Playback finished successfully.")

    except Exception as e:
        # 如果触发了 Reflex，这里会捕获到具体的异常信息
        print(f"[{arm_name}] EXCEPTION during control: {e}")
        abort_event.set()
        
    finally:
        # 无论成功还是报错，都要确保停止控制，释放锁
        try:
            robot.stop()
        except:
            pass

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