from __future__ import annotations

import contextlib
from dataclasses import dataclass
import inspect
import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from scipy.spatial.transform import Rotation
import torch


MANO_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_MANO_FINGER_BONES = {
    "thumb": {"proximal": 13, "middle": 14, "distal": 15, "fingertip": 15},
    "index": {"proximal": 1, "middle": 2, "distal": 3, "fingertip": 3},
    "middle": {"proximal": 4, "middle": 5, "distal": 6, "fingertip": 6},
    "ring": {"proximal": 10, "middle": 11, "distal": 12, "fingertip": 12},
    "pinky": {"proximal": 7, "middle": 8, "distal": 9, "fingertip": 9},
}


@dataclass(frozen=True)
class ManoModelConfig:
    mano_root: Path
    mano_side: str = "right"
    device: str = "cpu"
    flat_hand_mean: bool = False


@dataclass(frozen=True)
class ManoFitResult:
    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    transl: np.ndarray
    vertices: np.ndarray
    joints: np.ndarray
    loss: float
    residual_rms: float
    metadata: dict[str, Any]


class ManoModel:
    def __init__(self, config: ManoModelConfig) -> None:
        self.config = config
        self._validate_config()
        self._device = torch.device(config.device)
        self._mano = self._create_mano_model()

    def forward(
        self,
        *,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        betas: np.ndarray,
        transl: np.ndarray,
    ) -> dict[str, np.ndarray]:
        global_orient_t = torch.as_tensor(
            np.asarray(global_orient, dtype=np.float32).reshape(1, 3),
            device=self._device,
        )
        hand_pose_t = torch.as_tensor(
            np.asarray(hand_pose, dtype=np.float32).reshape(1, 45),
            device=self._device,
        )
        betas_t = torch.as_tensor(
            np.asarray(betas, dtype=np.float32).reshape(1, 10),
            device=self._device,
        )
        transl_t = torch.as_tensor(
            np.asarray(transl, dtype=np.float32).reshape(1, 3),
            device=self._device,
        )
        output = self._mano(
            global_orient=global_orient_t,
            hand_pose=hand_pose_t,
            betas=betas_t,
            transl=transl_t,
        )
        return {
            "vertices": output.vertices.detach().cpu().numpy().astype(np.float32),
            "joints": output.joints.detach().cpu().numpy().astype(np.float32),
        }

    def fit_initial_orient(self, rotation_matrix: np.ndarray) -> np.ndarray:
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32).reshape(3, 3)
        return Rotation.from_matrix(rotation_matrix).as_rotvec().astype(np.float32)

    @property
    def faces(self) -> np.ndarray:
        return np.asarray(self._mano.faces, dtype=np.int64)

    @property
    def skinning_weights(self) -> np.ndarray:
        return self._mano.lbs_weights.detach().cpu().numpy().astype(np.float32)

    def finger_vertex_indices(
        self,
        segments: tuple[str, ...] = ("proximal", "middle", "distal", "fingertip"),
    ) -> dict[str, np.ndarray]:
        unknown = sorted(
            set(segments).difference(("proximal", "middle", "distal", "fingertip"))
        )
        if unknown:
            raise ValueError(f"unknown MANO finger segments: {unknown}")
        dominant = self.skinning_weights.argmax(axis=1)
        regions = {}
        for finger in MANO_FINGER_NAMES:
            bones = {_MANO_FINGER_BONES[finger][segment] for segment in segments}
            indices = np.flatnonzero(np.isin(dominant, list(bones)))
            if indices.size == 0:
                raise RuntimeError(f"MANO semantic region for {finger} is empty")
            regions[finger] = indices
        return regions

    def _create_mano_model(self) -> Any:
        model = create_mano_model_quietly(
            model_path=(self.config.mano_root / "models").as_posix(),
            is_rhand=self.config.mano_side == "right",
            use_pca=False,
            flat_hand_mean=bool(self.config.flat_hand_mean),
            batch_size=1,
        )
        return model.to(self._device)

    def _validate_config(self) -> None:
        if self.config.mano_side not in ("right", "left"):
            raise ValueError("mano_side must be 'right' or 'left'")
        if self.config.device != "cpu" and not self.config.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or start with 'cuda'")
        mano_root = Path(self.config.mano_root).resolve()
        if not mano_root.is_dir():
            raise FileNotFoundError(f"MANO root does not exist: {mano_root}")
        mano_model = mano_root / "models" / f"MANO_{self.config.mano_side.upper()}.pkl"
        if not mano_model.is_file():
            raise FileNotFoundError(f"missing MANO model file: {mano_model}")


def patch_numpy_for_chumpy() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def create_mano_model_quietly(**kwargs: Any) -> Any:
    # chumpy 0.70 still calls inspect.getargspec, which was removed in
    # Python 3.11.  Its only use is reading the positional ``args`` field, so
    # FullArgSpec provides the compatible interface on modern Python versions.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    patch_numpy_for_chumpy()
    import smplx

    with contextlib.redirect_stdout(_SmplxShapeWarningFilter(sys.stdout)):
        return smplx.MANO(**kwargs)


def flat_hand_mean_from_contract(
    payload: dict[str, Any], *, default: bool = False
) -> bool:
    if not isinstance(default, bool):
        raise TypeError("default flat_hand_mean must be boolean")
    contract = payload.get("mano_contract")
    if contract is None:
        return default
    if not isinstance(contract, dict):
        raise TypeError("checkpoint mano_contract must be a mapping")
    expected = {
        "pose_dimension": 45,
        "pose_representation": "full_joint_axis_angle",
        "use_pca": False,
    }
    for key, expected_value in expected.items():
        if contract.get(key) != expected_value:
            raise ValueError(
                f"unsupported mano_contract.{key}: {contract.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    value = contract.get("flat_hand_mean", default)
    if not isinstance(value, bool):
        raise TypeError("mano_contract.flat_hand_mean must be boolean")
    return value


def mano_contract_from_dataset_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    dataset_format = metadata.get("dataset_format")
    return {
        "pose_dimension": 45,
        "pose_representation": "full_joint_axis_angle",
        "use_pca": False,
        "flat_hand_mean": dataset_format
        == "grasp_generator_grab_robodex_world_views_packed_v1",
    }


class _SmplxShapeWarningFilter:
    _SUPPRESSED = "WARNING: You are using a MANO model, with only 10 shape coefficients."

    def __init__(self, wrapped: TextIO) -> None:
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        if self._SUPPRESSED in text:
            return len(text)
        return self._wrapped.write(text)

    def flush(self) -> None:
        self._wrapped.flush()
