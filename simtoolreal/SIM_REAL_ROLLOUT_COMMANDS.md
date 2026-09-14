# SimToolReal: 真实初始姿态仿真与真实回放

本流程以本次设置的真实机器人 pose 为初始条件。`real_snapshot.json` 记录 FR3 7 维关节、Wuji 20 维关节、物体和目标 pose。Isaac rollout 在 policy 的第一帧 observation 前，将这些值写入仿真；训练中的 Isaac 默认 pose 不作为本次 rollout 的初始状态。

每次真实回放结束后，必须将 FR3 和 Wuji Hand 2 一起复位到已确认正确的 `calibration/runs/real_snapshot.json`，验证通过后再开始下一次回放。

## 0. 安全和约束

服务器：`pair1@192.168.50.13:/home/pair1/real-exp`。客户端：`/home/landau/real-exp`。

- 真实运动前确认 FR3 为 FCI 模式、工作区清空、急停可用。
- 不要同时运行 `policy_executor.py` 与 `replay_rollout_real.py`。
- `scripts/move_to_target_ee.sh --right --hand` 会启动自己的 Wuji worker 并占用 `5562`。它不能与本流程的 `start_client.sh` 同时运行。
- 本流程用 `return_to_snapshot_pose.py` 经由已经启动的 `5562` worker 复位手部，不需要停止硬件栈。复位 target 的完整 27 维值必须通过训练 URDF 和 FR3 hardware command limits 检查。

## 1. 启动 FoundationPose++ 和 policy 服务

在服务器新终端执行并保持运行：

```bash
ssh -Y pair1@192.168.50.13
cd /home/pair1/real-exp
conda activate pose

./simtoolreal/scripts/start_server.sh \
  --server-ip 192.168.50.13 \
  --foundationpose-mesh libs/FoundationPose-plus-plus/test/mesh/hammer.stl
```

确认 FoundationPose++ 已开始发布 live object pose，且 `5570` 可用。此步骤不发送真实机器人命令。

## 2. 启动 FR3/Wuji ROS、bridge 和 hand worker

在客户端新终端执行并保持运行：

```bash
cd /home/landau/real-exp

PATH="$HOME/anaconda3/bin:$PATH" \
./simtoolreal/scripts/start_client.sh \
  --server-ip 192.168.50.13 \
  --ros-distro humble \
  --ros-domain-id 73 \
  --local-bridge
```

确认状态流 `5555`、FR3 command bridge `5556`、Wuji worker `5562` 和 20 维手部 telemetry 均正常。

### 2.1 设置并采集本次初始 pose

将 FR3 和 Wuji 手设置为本次实验需要的初始 pose。若确实使用 `scripts/move_to_target_ee.sh --right --hand` 设置手部，先停止 `start_client.sh`，执行该脚本并确认完成，再重启本节的 `start_client.sh`。

待机器人、手部、物体稳定后，在客户端采集 snapshot：

```bash
cd /home/landau/real-exp

python3 simtoolreal/capture_real_snapshot.py \
  --server-ip 192.168.50.13 \
  --state-connect tcp://127.0.0.1:5555 \
  --world-from-camera calibration/generated/simtoolreal_policy_frame/world_from_l515_policy.json \
  --world-from-robot calibration/generated/simtoolreal_policy_frame/world_from_robot_policy.json \
  --goal-pose calibration/generated/simtoolreal_policy_frame/goal_policy.json \
  --pose-frame camera \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --mesh-path libs/FoundationPose-plus-plus/test/mesh/hammer.stl \
  --mesh-scale 0.001 \
  --object-scales 1,1,1 \
  --output calibration/runs/real_snapshot.json
```

只在显示 `Wrote read-only replay snapshot` 后继续。上传快照：

```bash
scp calibration/runs/real_snapshot.json \
  pair1@192.168.50.13:/home/pair1/real-exp/calibration/runs/real_snapshot.json
```

从这里到首次真实回放检查前，不要移动 FR3、Wuji、物体或目标。

## 3. Isaac Sim：以真实 pose 初始化并生成 rollout

在服务器执行。先运行 50 步；确认无异常后使用新 `RUN_ID` 和 `--steps 600` 生成正式轨迹。

```bash
cd /home/pair1/real-exp

RUN_ID=real_pose_$(date +%Y%m%d_%H%M%S)
V=libs/SimToolReal-Franka-Wuji2/.venv_isaacsim

PYTHONPATH=simtoolreal:$PWD/libs/SimToolReal-Franka-Wuji2 \
"$V/bin/python" simtoolreal/run_sim_rollout.py \
  --snapshot calibration/runs/real_snapshot.json \
  --config libs/SimToolReal-Franka-Wuji2/pretrained_policy/config.yaml \
  --checkpoint libs/SimToolReal-Franka-Wuji2/pretrained_policy/model.pth \
  --upstream-root libs/SimToolReal-Franka-Wuji2 \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --mesh libs/FoundationPose-plus-plus/test/mesh/hammer.stl \
  --steps 600 \
  --rate 60 \
  --seed 0 \
  --device cuda \
  --out outputs/simtoolreal_rollouts/$RUN_ID

echo "RUN_ID=$RUN_ID"
```

验证：

```bash
"$V/bin/python" - "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

p = Path('outputs/simtoolreal_rollouts') / sys.argv[1]
m = json.loads((p / 'metadata.json').read_text())
d = np.load(p / 'rollout.npz')
assert m['simulation_backend'] == 'isaac'
assert m['recorded_steps'] == m['requested_steps']
assert d['target'].shape == (m['recorded_steps'], 27)
assert np.isfinite(d['target']).all()
assert not d['terminated'].any() and not d['truncated'].any()
print('rollout validation: PASS')
PY
```

复制 rollout 到客户端。之后至第一次真实回放检查前，不要移动 FR3、Wuji、物体或目标：

```bash
scp -r /home/pair1/real-exp/outputs/simtoolreal_rollouts/$RUN_ID \
  landau@client:/home/landau/real-exp/outputs/simtoolreal_rollouts/
```

## 4. 真实机器人：检查、回放、复位

以下命令在客户端执行。

### 4.1 初始状态 dry-run

`--move-to-initial` 只比较当前 27 维状态与 snapshot，绝不移动机器人：

```bash
cd /home/landau/real-exp

python3 simtoolreal/replay_rollout_real.py \
  --rollout outputs/simtoolreal_rollouts/$RUN_ID \
  --state-connect tcp://127.0.0.1:5555 \
  --arm-command-connect tcp://127.0.0.1:5556 \
  --hand-command-address tcp://127.0.0.1:5562 \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --move-to-initial \
  --dry-run \
  --log-dir outputs/real_replays/${RUN_ID}_dry_run
```

只有 `initial max error < 0.06` 且最终输出 `"execute": false` 时才能继续。若失败，运行第 4.3 节的复位命令；不要将机器人对齐到训练 pose。

### 4.2 执行一次真实回放

确认安全后执行：

```bash
python3 simtoolreal/replay_rollout_real.py \
  --rollout outputs/simtoolreal_rollouts/$RUN_ID \
  --state-connect tcp://127.0.0.1:5555 \
  --arm-command-connect tcp://127.0.0.1:5556 \
  --hand-command-address tcp://127.0.0.1:5562 \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --move-to-initial \
  --confirm \
  --execute \
  --log-dir outputs/real_replays/$RUN_ID
```

按 `Ctrl-C` 会停止后续 target 发送。无论正常结束或中止，都执行下一节复位。

### 4.3 每次回放后：FR3 和 Wuji 一起复位到当前正确的真实初始 pose

先进行只读 dry-run：

```bash
python3 simtoolreal/return_to_snapshot_pose.py \
  --snapshot calibration/runs/real_snapshot.json \
  --state-connect tcp://127.0.0.1:5555 \
  --arm-command-connect tcp://127.0.0.1:5556 \
  --hand-command-address tcp://127.0.0.1:5562 \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --dry-run
```

确认目标与工作区后执行。该工具用 30 秒平滑插补 FR3，并以 20 Hz 持续向 Wuji worker 发送此正确 snapshot 的 20 维手部关节目标，最后读取 `5555` 实测状态检查 arm/hand 误差：

```bash
python3 simtoolreal/return_to_snapshot_pose.py \
  --snapshot calibration/runs/real_snapshot.json \
  --state-connect tcp://127.0.0.1:5555 \
  --arm-command-connect tcp://127.0.0.1:5556 \
  --hand-command-address tcp://127.0.0.1:5562 \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --duration 30 \
  --hold 5 \
  --confirm \
  --execute
```

只有输出 `return_to_snapshot: PASS` 后，才开始下一次 dry-run 或真实回放。若手部未达到默认 `--hand-tolerance 0.06`，不要放宽阈值或继续回放；检查 Wuji telemetry、机械限位、障碍和手部状态。

## 5. 计算 sim--real gap

真实回放日志保存后：

```bash
cd /home/landau/real-exp

python3 simtoolreal/compare_rollouts.py \
  --simulation outputs/simtoolreal_rollouts/$RUN_ID \
  --real outputs/real_replays/$RUN_ID \
  --robot-urdf simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
  --out outputs/simtoolreal_comparison/$RUN_ID
```

输出 `comparison_report.json` 和 `aligned_trajectories.npz`。

## 6. 停止顺序

1. `Ctrl-C` 停止回放或复位工具。
2. `Ctrl-C` 停止客户端 `start_client.sh`。
3. `Ctrl-C` 停止服务器 `start_server.sh`。
