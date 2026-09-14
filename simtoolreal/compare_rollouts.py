#!/usr/bin/env python3
"""Align simulation and real replay logs and report joint/cartesian errors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from kinematics import PolicyKinematics

def stats(x):
    x=np.asarray(x,float); return {"rmse":float(np.sqrt(np.mean(x*x))),"max_abs":float(np.max(np.abs(x))),"final_abs":float(np.max(np.abs(x[-1])))}
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--simulation",type=Path,required=True); p.add_argument("--real",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--robot-urdf",type=Path)
    a=p.parse_args(); s=np.load(a.simulation/"rollout.npz"); r=np.load(a.real/"real_replay.npz"); n=min(len(s["time_s"]),len(r["time_s"]));
    if n==0: raise SystemExit("empty trajectories")
    simq=np.asarray(s["joint_position"][:n]); realq=np.asarray(r["joint_position"][:n]); target=np.asarray(s["target"][:n]);
    if not np.all(np.isfinite(simq)) or not np.all(np.isfinite(target)):
        raise SystemExit("simulation trajectory contains non-finite values")
    if not np.all(np.isfinite(realq)):
        raise SystemExit("real trajectory contains non-finite values (dry-run logs cannot be compared)")
    err=realq-target
    report={"format":"simtoolreal_comparison_v1","frames":n,"simulation_frames":len(s["time_s"]),"real_frames":len(r["time_s"]),"joint_target_vs_real":stats(err),"joint_sim_vs_real":stats(realq-simq),"joint_target_vs_sim":stats(target-simq),"per_joint_rmse":np.sqrt(np.mean(err*err,axis=0)).tolist()}
    if "time_ns" in r and len(r["time_ns"]) >= n:
        wall = (np.asarray(r["time_ns"][:n], dtype=np.float64) - float(r["time_ns"][0])) * 1e-9
        expected = np.asarray(s["time_s"][:n], dtype=np.float64)
        timing_error = wall - expected
        report["timing"] = {
            "real_elapsed_s": float(wall[-1]),
            "simulation_elapsed_s": float(expected[-1]),
            "offset_rmse_ms": float(np.sqrt(np.mean(timing_error * timing_error)) * 1e3),
            "offset_max_abs_ms": float(np.max(np.abs(timing_error)) * 1e3),
            "real_interval_mean_ms": float(np.mean(np.diff(wall)) * 1e3) if n > 1 else 0.0,
            "real_interval_max_ms": float(np.max(np.diff(wall)) * 1e3) if n > 1 else 0.0,
        }
    if a.robot_urdf:
        meta=json.loads((a.simulation/"metadata.json").read_text()); wf=np.asarray(meta["world_from_robot"]); fk=PolicyKinematics(a.robot_urdf); sp=[]; rp=[]
        for x,y in zip(simq,realq): sp.append(fk.evaluate(x,wf)[0]); rp.append(fk.evaluate(y,wf)[0])
        report["palm_position_target_vs_real_m"]=stats(np.asarray([fk.evaluate(x,wf)[0] for x in target])-np.asarray(rp)); report["palm_position_sim_vs_real_m"]=stats(np.asarray(sp)-np.asarray(rp))
    a.out.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out/"aligned_trajectories.npz",time_s=np.asarray(s["time_s"][:n]),simulation_state=simq,real_state=realq,target=target,error=err); (a.out/"comparison_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
