#!/usr/bin/env python3
"""Safely replay recorded absolute SimToolReal targets on the real robot."""
from __future__ import annotations
import argparse, json, signal, time
from pathlib import Path
import numpy as np
import zmq
from policy_contract import hardware_command_limits, load_joint_limits
from policy_executor import bridge_state

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--rollout",type=Path,required=True)
    p.add_argument("--state-connect",default="tcp://127.0.0.1:5555"); p.add_argument("--arm-command-connect",default="tcp://127.0.0.1:5556")
    p.add_argument("--hand-command-address",default="tcp://127.0.0.1:5562"); p.add_argument("--robot-urdf",type=Path,required=True)
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--execute",action="store_true"); p.add_argument("--confirm",action="store_true")
    p.add_argument("--move-to-initial",action="store_true"); p.add_argument("--position-tolerance",type=float,default=.06)
    p.add_argument("--max-target-step",type=float,default=.5,help="maximum adjacent target change in radians")
    p.add_argument("--log-dir",type=Path,required=True)
    a=p.parse_args();
    if a.execute and (not a.confirm or a.dry_run): p.error("--execute requires --confirm and must not use --dry-run")
    data=np.load(a.rollout/"rollout.npz"); targets=np.asarray(data["target"],float); ts=np.asarray(data["time_s"],float)
    meta_path = a.rollout / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        backend = str(meta.get("simulation_backend", ""))
        if backend != "isaac" and a.execute:
            p.error(f"refusing real replay of non-physics rollout backend={backend!r}; generate an Isaac rollout first")
    if targets.ndim!=2 or targets.shape[1]!=27 or not np.all(np.isfinite(targets)): p.error("target must be finite [N,27]")
    if ts.ndim != 1 or len(ts) != len(targets) or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) < 0):
        p.error("time_s must be finite, one-dimensional, and nondecreasing")
    if a.max_target_step <= 0 or (len(targets) > 1 and float(np.max(np.abs(np.diff(targets, axis=0)))) > a.max_target_step):
        p.error(f"adjacent target step exceeds --max-target-step={a.max_target_step}")
    lower,upper=load_joint_limits(a.robot_urdf); cl,cu=hardware_command_limits(lower,upper)
    if np.any(targets<cl) or np.any(targets>cu): p.error("rollout target exceeds hardware limits")
    a.log_dir.mkdir(parents=True,exist_ok=True); log_q=[]; log_t=[]; ctx=zmq.Context(); ss=ctx.socket(zmq.SUB); ss.setsockopt(zmq.SUBSCRIBE,b""); ss.setsockopt(zmq.CONFLATE,1); ss.connect(a.state_connect)
    arm=hand=None
    if a.execute:
        arm=ctx.socket(zmq.PUSH); arm.setsockopt(zmq.SNDHWM,1); arm.connect(a.arm_command_connect)
        hand=ctx.socket(zmq.PUSH); hand.setsockopt(zmq.SNDHWM,1); hand.connect(a.hand_command_address)
    stop=False
    def halt(*_):
        nonlocal stop; stop=True
    signal.signal(signal.SIGINT,halt); signal.signal(signal.SIGTERM,halt)
    def read_state(timeout=5.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if ss.poll(100):
                q,_,stamp=bridge_state(ss.recv_pyobj()); return q,stamp
        raise TimeoutError("state stream timeout")
    initial=None
    if a.move_to_initial:
        initial=np.asarray(json.loads((a.rollout/"metadata.json").read_text())["initial_joint_position"])
        q,_=read_state(); initial_error=float(np.max(np.abs(q-initial)))
        print("initial max error", initial_error)
        if initial_error > a.position_tolerance:
            raise RuntimeError("real robot is not at snapshot initial pose; move it with align_initial_pose.py, then retry")
    started=time.monotonic();
    for target,t in zip(targets,ts):
        if stop: break
        wait=started+float(t)-time.monotonic()
        if wait>0: time.sleep(wait)
        if a.dry_run:
            q = np.full(27, np.nan)
        else:
            q,stamp=read_state(timeout=2.)
        log_q.append(q); log_t.append(time.time_ns())
        if a.execute:
            arm.send_pyobj({"timestamp":time.time(),"right_joint_target":target[:7].tolist()}); hand.send_pyobj(target[7:].tolist())
    np.savez_compressed(a.log_dir/"real_replay.npz",time_ns=np.asarray(log_t),joint_position=np.asarray(log_q),target=targets[:len(log_q)],time_s=ts[:len(log_q)])
    (a.log_dir/"metadata.json").write_text(json.dumps({"format":"simtoolreal_real_replay_v1","execute":a.execute,"frames":len(log_q),"rollout":str(a.rollout) },indent=2)+"\n")
    for s in (ss,arm,hand):
        if s is not None: s.close(0)
    ctx.term(); print(json.dumps({"frames":len(log_q),"execute":a.execute},indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
