import argparse
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    print("错误: 缺少 pyarrow 库。请在环境中运行 `pip install pyarrow`。")
    sys.exit(1)

# ================= 配置常量 =================
# 夹爪在 16 维数组中的绝对索引
GRIPPER_INDEX_LEFT = 7
GRIPPER_INDEX_RIGHT = 15

# 根据 validate_dataset.py 设定的标准
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
TOLERANCE = 1e-5  # 浮点数容差

# 状态枚举
STATUS_OK = 0          # 正常的 0 或 1 (在容差范围内)
STATUS_NON_BINARY = 1  # 0 到 1 之间的中间值 (非二值)
STATUS_OUT_OF_BOUNDS = 2 # 小于 0 或 大于 1 (越界)
# ============================================

def evaluate_gripper_value(val: float) -> tuple[int, str]:
    """
    评估夹爪数值，判断其属于哪种状态。
    返回: (状态码, 描述字符串)
    """
    if abs(val - GRIPPER_MIN) <= TOLERANCE:
        return STATUS_OK, "CLOSE (0.0)"
    elif abs(val - GRIPPER_MAX) <= TOLERANCE:
        return STATUS_OK, "OPEN (1.0)"
    elif val < GRIPPER_MIN - TOLERANCE or val > GRIPPER_MAX + TOLERANCE:
        return STATUS_OUT_OF_BOUNDS, f"越界 ({val:.6f})"
    else:
        return STATUS_NON_BINARY, f"非二值中间态 ({val:.6f})"

def main():
    parser = argparse.ArgumentParser(description="检查 LeRobot 数据集中夹爪的二值状态 (Action & State)")
    parser.add_argument(
        "--dataset-root", 
        type=str, 
        default=".", 
        help="数据集根目录路径 (默认: 当前目录)"
    )
    parser.add_argument(
        "--max-print", 
        type=int, 
        default=10, 
        help="最多打印多少条越界信息 (防止刷屏，默认: 10)"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    data_dir = dataset_root / "data"

    if not data_dir.exists():
        print(f"错误: 找不到数据目录 {data_dir}。请确认 --dataset-root 路径是否正确。")
        sys.exit(1)

    # 查找所有的 parquet 文件
    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        print(f"错误: 在 {data_dir} 下没有找到任何 parquet 文件。")
        sys.exit(1)

    print(f"🚀 开始检查数据集: {dataset_root}")
    print(f"📂 找到 {len(parquet_files)} 个 Parquet 文件，正在读取...")
    print(f"⚙️  检查标准: Min={GRIPPER_MIN}, Max={GRIPPER_MAX}, 容差={TOLERANCE}")
    print("-" * 60)

    # 统计数据
    stats = {
        "total_frames": 0,
        "action": {"ok": 0, "non_binary": 0, "out_of_bounds": 0},
        "state": {"ok": 0, "non_binary": 0, "out_of_bounds": 0}
    }
    
    anomaly_records = []

    # 遍历所有 parquet 文件
    for pf in parquet_files:
        table = pq.read_table(pf)
        for row in table.to_pylist():
            stats["total_frames"] += 1
            
            ep_idx = row.get("episode_index", "?")
            frame_idx = row.get("frame_index", "?")
            action = row.get("action", [])
            state = row.get("observation.state", [])

            # --- 检查 Action (动作指令) ---
            if len(action) >= 16:
                act_l_val = action[GRIPPER_INDEX_LEFT]
                act_r_val = action[GRIPPER_INDEX_RIGHT]
                
                act_l_code, act_l_desc = evaluate_gripper_value(act_l_val)
                act_r_code, act_r_desc = evaluate_gripper_value(act_r_val)
                
                # 统计 Action
                if act_l_code == STATUS_OK and act_r_code == STATUS_OK:
                    stats["action"]["ok"] += 1
                else:
                    if act_l_code == STATUS_OUT_OF_BOUNDS or act_r_code == STATUS_OUT_OF_BOUNDS:
                        stats["action"]["out_of_bounds"] += 1
                        # 【最小修改点 1】只在严重越界时，才将记录加入打印列表
                        anomaly_records.append(
                            f"[Action 严重越界] Ep:{ep_idx} Frame:{frame_idx} | 左:{act_l_desc}, 右:{act_r_desc}"
                        )
                    else:
                        stats["action"]["non_binary"] += 1

            # --- 检查 State (物理状态) ---
            if len(state) >= 16:
                st_l_val = state[GRIPPER_INDEX_LEFT]
                st_r_val = state[GRIPPER_INDEX_RIGHT]
                
                st_l_code, st_l_desc = evaluate_gripper_value(st_l_val)
                st_r_code, st_r_desc = evaluate_gripper_value(st_r_val)
                
                # 统计 State
                if st_l_code == STATUS_OK and st_r_code == STATUS_OK:
                    stats["state"]["ok"] += 1
                else:
                    if st_l_code == STATUS_OUT_OF_BOUNDS or st_r_code == STATUS_OUT_OF_BOUNDS:
                        stats["state"]["out_of_bounds"] += 1
                        # 【最小修改点 2】只在严重越界时，才将记录加入打印列表
                        anomaly_records.append(
                            f"[State  严重越界] Ep:{ep_idx} Frame:{frame_idx} | 左:{st_l_desc}, 右:{st_r_desc}"
                        )
                    else:
                        stats["state"]["non_binary"] += 1

    # ================= 打印结果 =================
    if anomaly_records:
        # 【最小修改点 3】修改了这里的提示文案，明确是“严重越界”
        print(f"⚠️ 发现严重越界！以下是前 {min(args.max_print, len(anomaly_records))} 条越界记录：")
        for record in anomaly_records[:args.max_print]:
            print("  " + record)
        if len(anomaly_records) > args.max_print:
            print(f"  ... (省略其余 {len(anomaly_records) - args.max_print} 条越界记录)")
    else:
        print("✅ 完美！没有发现任何严重越界的数据。")

    print("\n" + "=" * 20 + " 📊 统计汇总 " + "=" * 20)
    print(f"总帧数: {stats['total_frames']}")
    print("\n【Action (动作指令) 统计】 - 理论上应 100% 正常")
    print(f"  ✅ 正常二值 (0或1): {stats['action']['ok']} 帧")
    print(f"  ⚠️ 非二值中间态  : {stats['action']['non_binary']} 帧")
    print(f"  ❌ 严重越界      : {stats['action']['out_of_bounds']} 帧")
    
    print("\n【State (物理状态) 统计】 - 如果传感器有延迟或连续闭合过程，可能会出现非二值")
    print(f"  ✅ 正常二值 (0或1): {stats['state']['ok']} 帧")
    print(f"  ⚠️ 非二值中间态  : {stats['state']['non_binary']} 帧")
    print(f"  ❌ 严重越界      : {stats['state']['out_of_bounds']} 帧")
    print("=" * 54)

if __name__ == "__main__":
    main()