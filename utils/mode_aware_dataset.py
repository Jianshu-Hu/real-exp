"""Mode-specific, chunk-anchored policy views over recorded robot telemetry."""
from __future__ import annotations
import copy
from typing import Any
import numpy as np
import torch
from utils.fr3_kinematics import ee_delta, ee_state_to_matrix
from utils.trajectory_metadata import ARM_JOINT_DIM, EE_ACTION_DIM, EE_STATE_DIM, HAND_JOINT_DIM, TRAJECTORY_CONFIG_SCHEMA_VERSION, validate_trajectory_config

def normalize_training_mode(mode: str | None, default: str) -> str:
    value = default if mode is None else str(mode).strip().lower()
    if value in {"ee", "pose", "end_effector_pose", "end-effector"}: value = "end_effector"
    if value not in {"joint", "end_effector"}: raise ValueError(f"Unsupported training state/action mode {value!r}.")
    return value

def _ee_dim(config): return {"arm": 0, "gripper": 1, "hand": HAND_JOINT_DIM}[config["end_effector"]]
def _feature_dim(info, key):
    try: return int(info["features"][key]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc: raise ValueError(f"Dataset is missing a valid {key} feature.") from exc

def mode_trajectory_config(source_config, mode, *, state_dim, action_dim):
    mode = normalize_training_mode(mode, str(source_config.get("state_action_mode", "joint")))
    n, ee = len(source_config["arms"]), _ee_dim(source_config)
    if int(source_config.get("schema_version", 1)) == 1 and mode == "end_effector" and state_dim == action_dim and state_dim in {6 * n, (6 + ee) * n}:
        policy_dim = (6 + ee) * n
        config = dict(source_config)
        config.update(robot_state_dim=policy_dim, action_dim=policy_dim, state_action_mode=mode, state_representation="end_effector_pose", action_representation="delta_end_effector_pose")
        return validate_trajectory_config(config, policy_dim, policy_dim, source="selected training mode")
    if int(source_config.get("schema_version", 1)) == 1 and mode == "end_effector" and state_dim == action_dim and state_dim in {6 * n, (6 + ee) * n}:
        config = dict(source_config)
        config.update(robot_state_dim=state_dim, action_dim=action_dim, state_action_mode=mode, state_representation="end_effector_pose", action_representation="delta_end_effector_pose")
        return validate_trajectory_config(config, state_dim, action_dim, source="selected training mode")
    source_state_expected = (ARM_JOINT_DIM if mode == "joint" else EE_STATE_DIM) * n
    source_action_expected = (ARM_JOINT_DIM if mode == "joint" else EE_ACTION_DIM) * n
    accepted_action_dims = {source_action_expected}
    if mode == "end_effector":
        # The wrapper deliberately consumes absolute target_ee_pose (18D) and
        # derives the 12D chunk delta against one shared anchor.
        accepted_action_dims.add(EE_STATE_DIM * n)
    if state_dim != source_state_expected or action_dim not in accepted_action_dims:
        raise ValueError(f"Cannot train {mode} mode from {state_dim}/{action_dim}; expected state {source_state_expected} and action one of {sorted(accepted_action_dims)}.")
    state_dim_out = ((ARM_JOINT_DIM if mode == "joint" else EE_STATE_DIM) + ee) * n
    action_dim_out = ((ARM_JOINT_DIM if mode == "joint" else EE_ACTION_DIM) + ee) * n
    config = dict(source_config)
    config.update(schema_version=TRAJECTORY_CONFIG_SCHEMA_VERSION, robot_state_dim=state_dim_out, action_dim=action_dim_out, state_action_mode=mode, state_representation="joint" if mode == "joint" else "end_effector_position_rotation_6d", action_representation="delta_joint_position" if mode == "joint" else "delta_end_effector_position_rotation_vector", delta_alignment="chunk_anchor")
    return validate_trajectory_config(config, state_dim_out, action_dim_out, source="selected training mode")

def mode_action_config(source_config, mode, trajectory):
    result = dict(source_config)
    arm_rep = "delta_joint_position" if mode == "joint" else "delta_end_effector_position_rotation_vector"
    if int(trajectory.get("schema_version", 2)) == 1:
        arm_rep = "absolute_joint_position" if mode == "joint" else "delta_end_effector_pose"
    result.update(schema_version=int(trajectory.get("schema_version", TRAJECTORY_CONFIG_SCHEMA_VERSION)), action_dim=int(trajectory["action_dim"]), state_action_mode=trajectory["state_action_mode"], state_representation=trajectory["state_representation"], action_representation=trajectory["action_representation"], arm_action_representation=arm_rep, arm_action_definition="q_target[t+h]-q_measured[t]" if mode == "joint" else "base_spatial_delta(ee_measured[t],ee_target[t+h])", delta_alignment="chunk_anchor", chunk_anchor_definition="latest_generation_observation", ee_state_rotation_representation="rotation_6d_first_two_columns", ee_action_rotation_representation="rotation_vector", ee_action_rotation_frame="robot_base_spatial", ee_rotation_composition="R_target=Exp(rotvec)@R_anchor", transport_action_representation="absolute_target")
    return result

def _tensor(x, like=None):
    if isinstance(x, torch.Tensor): return x
    return torch.as_tensor(x, dtype=None if like is None else like.dtype, device=None if like is None else like.device)
def _blocks(x, n, extra):
    stride = ARM_JOINT_DIM + extra
    return [x[..., i*stride:(i+1)*stride] for i in range(n)]
def _joint_delta(anchor, target, n, extra):
    parts = []
    for ab, tb in zip(_blocks(anchor, n, extra), _blocks(target, n, extra)):
        parts.append(tb[..., :7] - ab[..., :7])
        if extra:
            parts.append(tb[..., 7:])
    return torch.cat(parts, -1)
def _ee_state(poses, joints, n, extra):
    if not extra: return poses
    parts = []
    blocks = _blocks(joints, n, extra)
    for i in range(n):
        parts.extend((poses[..., i * 9 : (i + 1) * 9], blocks[i][..., 7:]))
    return torch.cat(parts, -1)
def _ee_action(anchor, targets, joint_targets, n, extra):
    a = anchor.detach().cpu().numpy(); t = targets.detach().cpu().numpy().reshape(-1, targets.shape[-1]); out=[]
    for row in t:
        chunks=[]
        for i in range(n): chunks.append(ee_delta(ee_state_to_matrix(a[i*9:(i+1)*9]), ee_state_to_matrix(row[i*9:(i+1)*9])))
        out.append(np.concatenate(chunks))
    d = torch.as_tensor(np.asarray(out), dtype=targets.dtype, device=targets.device).reshape(*targets.shape[:-1], -1)
    if not extra: return d
    jb = _blocks(joint_targets,n,extra); parts=[]
    for i in range(n): parts.extend((d[...,i*6:(i+1)*6], jb[i][...,7:]))
    return torch.cat(parts,-1)
def _stats(values):
    x=np.asarray(values,dtype=np.float64)
    return {"min":x.min(0).astype(np.float32),"max":x.max(0).astype(np.float32),"mean":x.mean(0).astype(np.float32),"std":x.std(0).astype(np.float32),"count":np.asarray([len(x)]),"q01":np.quantile(x,.01,0).astype(np.float32),"q10":np.quantile(x,.1,0).astype(np.float32),"q50":np.quantile(x,.5,0).astype(np.float32),"q90":np.quantile(x,.9,0).astype(np.float32),"q99":np.quantile(x,.99,0).astype(np.float32)}

def _interleave_stats(arm_stats, joint_stats, n, extra, arm_dim):
    if not extra: return copy.deepcopy(arm_stats)
    out={}
    block=7+extra
    for key, value in arm_stats.items():
        a=np.asarray(value); j=np.asarray(joint_stats.get(key, value))
        if a.shape != (arm_dim*n,) or j.shape != (block*n,): out[key]=copy.deepcopy(value); continue
        parts=[]
        for i in range(n):
            parts.extend((a[i*arm_dim:(i+1)*arm_dim], j[i*block+7:(i+1)*block]))
        out[key]=np.concatenate(parts)
    return out

class ModeAwareDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mode):
        self.dataset=dataset; self.mode=normalize_training_mode(mode,"joint"); self.meta=copy.copy(dataset.meta); self.meta.info=copy.deepcopy(dataset.meta.info); self.meta.stats=copy.deepcopy(dataset.meta.stats)
        jd=_feature_dim(self.meta.info,"observation.joint_state"); candidates=[(n,e) for n in (1,2) for e in (0,1,20) if n*(7+e)==jd]
        if len(candidates)!=1: raise ValueError(f"Cannot infer robot layout from {jd}-value joint state.")
        self.n,self.extra=candidates[0]; self.state_key="observation.joint_state" if self.mode=="joint" else "observation.ee_pose"; self._legacy = False
        if self.mode == "joint":
            self.target_key = "action.target_joint"
            self._legacy = "action.delta_joint" not in self.meta.info["features"]
        else:
            self.target_key = "action.target_ee_pose" if "action.target_ee_pose" in self.meta.info["features"] else "action.delta_ee_pose"
            self._legacy = self.target_key == "action.delta_ee_pose"
        self._old_ee_compat = self.mode == "end_effector" and self._legacy and _feature_dim(self.meta.info, self.state_key) == _feature_dim(self.meta.info, self.target_key) and _feature_dim(self.meta.info, self.state_key) != EE_STATE_DIM * self.n
        for k in (self.state_key,self.target_key,"observation.joint_state","action.target_joint"):
            if k not in self.meta.info["features"]: raise ValueError(f"Dataset is missing required field {k!r}.")
        self._configure_windows()
        sd = ((6 + self.extra) * self.n) if self._old_ee_compat else ((7 if self.mode=="joint" else 9)+self.extra)*self.n
        ad = ((6 + self.extra) * self.n) if self._old_ee_compat else ((7 if self.mode=="joint" or self._legacy else 6)+self.extra)*self.n
        self.meta.info["features"]["observation.state"]=copy.deepcopy(self.meta.info["features"][self.state_key]); self.meta.info["features"]["action"]=copy.deepcopy(self.meta.info["features"][self.target_key]); self.meta.info["features"]["observation.state"]["shape"]=(sd,); self.meta.info["features"]["action"]["shape"]=(ad,)
        if self._old_ee_compat:
            self.meta.stats["observation.state"] = copy.deepcopy(self.meta.stats[self.state_key]); self.meta.stats["action"] = copy.deepcopy(self.meta.stats[self.target_key])
        elif self._legacy:
            self.meta.stats["observation.state"] = copy.deepcopy(self.meta.stats[self.state_key])
            self.meta.stats["action"] = copy.deepcopy(self.meta.stats[self.target_key])
        elif self.mode == "joint":
            self.meta.stats["observation.state"] = copy.deepcopy(self.meta.stats[self.state_key])
            self.meta.stats["action"] = self._chunk_stats()
        else:
            self.meta.stats["observation.state"] = _interleave_stats(self.meta.stats[self.state_key], self.meta.stats["observation.joint_state"], self.n, self.extra, 9)
            self.meta.stats["action"] = self._chunk_stats()
    def _configure_windows(self):
        d=getattr(self.dataset,"delta_timestamps",None)
        self.action_offsets=list(getattr(getattr(self.dataset,"reader",None),"delta_indices",{}).get("action",[0])) if d is not None else [0]
        if d is not None and "action" in d:
            u=dict(d); times=list(d["action"])
            for k in ("action.target_joint","action.target_ee_pose"): u[k]=times
            self.dataset.delta_timestamps=u
            if getattr(self.dataset,"reader",None) is not None:
                idx=dict(self.dataset.reader.delta_indices)
                for k in ("action.target_joint","action.target_ee_pose"): idx[k]=list(idx["action"])
                self.dataset.reader.delta_indices=idx
    def _chunk_stats(self):
        """Compute statistics on valid chunk targets when raw columns are available."""
        try:
            raw = self.dataset.hf_dataset
            state = np.asarray(raw[self.state_key], dtype=np.float32)
            joint_state = np.asarray(raw["observation.joint_state"], dtype=np.float32)
            target = np.asarray(raw[self.target_key], dtype=np.float32)
            joint_target = np.asarray(raw["action.target_joint"], dtype=np.float32)
            episodes = np.asarray(raw["episode_index"], dtype=np.int64)
        except Exception:
            return _interleave_stats(
                self.meta.stats.get("action.delta_joint" if self.mode == "joint" else "action.delta_ee_pose", self.meta.stats[self.target_key]),
                self.meta.stats["action.target_joint"], self.n, self.extra,
                7 if self.mode == "joint" else 6,
            )
        values=[]
        for i in range(len(state)):
            for offset in self.action_offsets:
                if offset < 0 or i + offset >= len(state) or episodes[i + offset] != episodes[i]: continue
                if self.mode == "joint": values.append(_joint_delta(torch.from_numpy(joint_state[i]), torch.from_numpy(joint_target[i+offset]), self.n, self.extra).numpy())
                else: values.append(_ee_action(torch.from_numpy(state[i]), torch.from_numpy(target[i+offset]), torch.from_numpy(joint_target[i+offset]), self.n, self.extra).numpy())
        return _stats(np.asarray(values)) if values else self.meta.stats["action"]
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index):
        item=dict(self.dataset[index]); st=_tensor(item[self.state_key]); js=_tensor(item["observation.joint_state"],st); anchor=st[-1] if st.ndim>1 else st; ja=js[-1] if js.ndim>1 else js; tgt=_tensor(item[self.target_key],st); jt=_tensor(item["action.target_joint"],st)
        if self._old_ee_compat:
            state = _ee_state(st, js, self.n, self.extra)
            parts=[]; blocks=_blocks(jt,self.n,self.extra)
            for i in range(self.n): parts.extend((tgt[..., i*6:(i+1)*6], blocks[i][...,7:]))
            action=torch.cat(parts,-1)
        elif self._legacy:
            state = st if self.mode == "joint" else _ee_state(st, js, self.n, self.extra)
            action = tgt
            if self.mode == "end_effector" and self.extra:
                # Legacy arm-only EE fields are composed with absolute
                # gripper/hand targets for compatibility with schema-v1 tests.
                parts=[]; blocks=_blocks(jt,self.n,self.extra)
                for i in range(self.n): parts.extend((tgt[..., i*6:(i+1)*6], blocks[i][...,7:]))
                action=torch.cat(parts,-1)
        else:
            state=js if self.mode=="joint" else _ee_state(st,js,self.n,self.extra); action=_joint_delta(ja,jt,self.n,self.extra) if self.mode=="joint" else _ee_action(anchor,tgt,jt,self.n,self.extra)
        pad=item.get(f"{self.target_key}_is_pad",item.get("action_is_pad")); pad=torch.zeros(action.shape[:-1],dtype=torch.bool) if pad is None else _tensor(pad).bool()
        if action.ndim>1: pad=pad|torch.as_tensor([o<0 for o in self.action_offsets],dtype=torch.bool)
        item["observation.state"]=state; item["action"]=action; item["action_is_pad"]=pad; return item
    def __getattr__(self,name): return getattr(self.dataset,name)
def adapt_dataset_for_mode(dataset,mode): return ModeAwareDataset(dataset,mode)
