import argparse
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ModuleNotFoundError:
    print("错误: 缺少 pyarrow 库。请在环境中运行 `pip install pyarrow`。")
    sys.exit(1)

# ================= 配置常量 =================
# 仅针对夹爪的索引进行裁剪，保护其他关节数据！
GRIPPER_INDEX_LEFT = 7
GRIPPER_INDEX_RIGHT = 15

GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
# TOLERANCE = 1e-05  # 增加容差常量
# ============================================

def clip_value(val: float) -> tuple[float, bool]:
    """将数值限制在 [MIN, MAX] 范围内，并返回是否发生了修改"""
    if val < GRIPPER_MIN:
        return GRIPPER_MIN, True
    elif val > GRIPPER_MAX:
        return GRIPPER_MAX, True
    return val, False

# def clip_value(val: float) -> tuple[float, bool]:
#     """带容差的修复逻辑：只修复超出 1e-5 的严重越界数据"""
#     # 只有当数值小于 -0.00001 时，才强制拉回 0.0
#     if val < GRIPPER_MIN - TOLERANCE:
#         return GRIPPER_MIN, True
#     # 只有当数值大于 1.00001 时，才强制拉回 1.0
#     elif val > GRIPPER_MAX + TOLERANCE:
#         return GRIPPER_MAX, True
    
#     # 处于 [-0.00001, 0.0] 或 [1.0, 1.00001] 之间的微小误差，不作修改，直接放行
#     return val, False

def main():
    parser = argparse.ArgumentParser(description="精准修复 LeRobot 数据集中夹爪的越界数据")
    parser.add_argument(
        "--dataset-root", 
        type=str, 
        required=True, 
        help="数据集根目录路径"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    data_dir = dataset_root / "data"

    if not data_dir.exists():
        print(f"❌ 错误: 找不到数据目录 {data_dir}。")
        sys.exit(1)

    # 兼容直接在 data 下或在 chunk-* 下的 parquet 文件
    parquet_files = list(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        parquet_files = list(data_dir.glob("*.parquet"))
        
    if not parquet_files:
        print(f"❌ 错误: 在 {data_dir} 下没有找到任何 parquet 文件。")
        sys.exit(1)

    print(f"🚀 开始修复数据集: {dataset_root}")
    print(f"📂 找到 {len(parquet_files)} 个 Parquet 文件，正在扫描并修复越界帧...")
    print(f"🛡️  安全机制: 仅裁剪索引 {GRIPPER_INDEX_LEFT} (左夹爪) 和 {GRIPPER_INDEX_RIGHT} (右夹爪)")
    print("-" * 60)

    total_fixed_state = 0
    total_fixed_action = 0

    for pf in parquet_files:
        # 读取 Parquet 文件和它的 Schema (为了保存时保持 LeRobot 的原始数据类型不变)
        table = pq.read_table(pf)
        schema = table.schema
        
        file_fixed_state = 0
        file_fixed_action = 0

        # --- 1. 修复 State (物理状态) ---
        if 'observation.state' in schema.names:
            state_col_idx = schema.get_field_index('observation.state')
            states = table.column('observation.state').to_pylist()
            
            for i in range(len(states)):
                state_arr = states[i]
                fixed_l = fixed_r = False
                
                if len(state_arr) > GRIPPER_INDEX_LEFT:
                    state_arr[GRIPPER_INDEX_LEFT], fixed_l = clip_value(state_arr[GRIPPER_INDEX_LEFT])
                if len(state_arr) > GRIPPER_INDEX_RIGHT:
                    state_arr[GRIPPER_INDEX_RIGHT], fixed_r = clip_value(state_arr[GRIPPER_INDEX_RIGHT])
                    
                if fixed_l or fixed_r:
                    states[i] = state_arr
                    file_fixed_state += 1
            
            # 如果有修改，替换原来的列
            if file_fixed_state > 0:
                new_state_array = pa.array(states, type=schema.field('observation.state').type)
                table = table.set_column(state_col_idx, 'observation.state', new_state_array)
                total_fixed_state += file_fixed_state

        # --- 2. 修复 Action (动作指令) --- 
        # (虽然你的 action 检查没问题，但为了以防万一，顺便加上)
        if 'action' in schema.names:
            action_col_idx = schema.get_field_index('action')
            actions = table.column('action').to_pylist()
            
            for i in range(len(actions)):
                action_arr = actions[i]
                fixed_l = fixed_r = False
                
                if len(action_arr) > GRIPPER_INDEX_LEFT:
                    action_arr[GRIPPER_INDEX_LEFT], fixed_l = clip_value(action_arr[GRIPPER_INDEX_LEFT])
                if len(action_arr) > GRIPPER_INDEX_RIGHT:
                    action_arr[GRIPPER_INDEX_RIGHT], fixed_r = clip_value(action_arr[GRIPPER_INDEX_RIGHT])
                    
                if fixed_l or fixed_r:
                    actions[i] = action_arr
                    file_fixed_action += 1
            
            # 如果有修改，替换原来的列
            if file_fixed_action > 0:
                new_action_array = pa.array(actions, type=schema.field('action').type)
                table = table.set_column(action_col_idx, 'action', new_action_array)
                total_fixed_action += file_fixed_action

        # --- 3. 覆写保存 ---
        if file_fixed_state > 0 or file_fixed_action > 0:
            pq.write_table(table, pf)
            print(f"  ✅ {pf.name}: 修复了 {file_fixed_state} 帧 State, {file_fixed_action} 帧 Action")

    # ================= 打印结果 =================
    print("-" * 60)
    if total_fixed_state == 0 and total_fixed_action == 0:
        print("🎉 扫描完毕！没有发现需要修复的越界数据。")
    else:
        print(f"🎉 修复完成！全局共修复了:")
        print(f"   - State  越界: {total_fixed_state} 帧")
        print(f"   - Action 越界: {total_fixed_action} 帧")
        print("💡 提示: 原 Parquet 文件已安全覆写，现在可以顺利进行 LeRobot 训练了！")

if __name__ == "__main__":
    main()