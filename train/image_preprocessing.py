from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms import functional as F

CAMERA_MASK_KEYS: tuple[str, str, str] = (
    "observation.images.cam_left",
    "observation.images.cam_front",
    "observation.images.cam_right",
)


@dataclass(frozen=True)
class ResizePadConfig:
    enabled: bool = True
    size: int = 224
    fill: float = 0.0


class ResizePadSquare:
    """Aspect-preserving resize followed by constant padding to a square."""

    def __init__(self, size: int = 224, fill: float = 0.0) -> None:
        if size <= 0:
            raise ValueError(f"Resize target must be positive. Got {size}.")
        self.size = int(size)
        self.fill = float(fill)

    def __call__(self, image: Tensor) -> Tensor:
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim < 3:
            raise ValueError(f"Expected image tensor with at least 3 dims (..., C, H, W). Got {image.shape}.")

        height = int(image.shape[-2])
        width = int(image.shape[-1])
        scale = self.size / max(height, width)
        resized_h = max(1, round(height * scale))
        resized_w = max(1, round(width * scale))

        image = F.resize(image, [resized_h, resized_w], antialias=True)

        pad_h = self.size - resized_h
        pad_w = self.size - resized_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        return F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


@dataclass(frozen=True)
class CameraMaskConfig:
    mode: str = "off"
    left_prob: float = 0.0
    front_prob: float = 0.0
    right_prob: float = 0.0
    fill: float = 0.0
    camera_keys: tuple[str, str, str] = CAMERA_MASK_KEYS

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def none_prob(self) -> float:
        return 1.0 - self.left_prob - self.front_prob - self.right_prob


def validate_camera_mask_config(config: CameraMaskConfig) -> None:
    if config.mode not in {"off", "single"}:
        raise ValueError(f"Unsupported camera mask mode '{config.mode}'. Expected 'off' or 'single'.")

    probabilities = {
        "left": config.left_prob,
        "front": config.front_prob,
        "right": config.right_prob,
    }
    for name, value in probabilities.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"camera mask {name} probability must be in [0, 1]. Got {value}.")

    if config.mode == "single" and config.none_prob < -1e-9:
        total = config.left_prob + config.front_prob + config.right_prob
        raise ValueError(
            "camera mask probabilities must sum to <= 1 in single mode. "
            f"Got left+front+right={total:.6g}."
        )


def require_camera_mask_keys(batch: dict[str, Any], config: CameraMaskConfig) -> None:
    missing_keys = [key for key in config.camera_keys if key not in batch]
    if missing_keys:
        raise KeyError(
            "Camera mask is enabled but the training batch is missing camera keys: "
            + ", ".join(missing_keys)
        )


def apply_camera_mask_to_batch(
    batch: dict[str, Any],
    config: CameraMaskConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Apply per-sample, at-most-one-camera masking to a training batch."""

    validate_camera_mask_config(config)
    if not config.enabled:
        return batch
    require_camera_mask_keys(batch, config)

    left_key, front_key, right_key = config.camera_keys
    first_image = batch[left_key]
    if not isinstance(first_image, torch.Tensor):
        first_image = torch.as_tensor(first_image)
        batch[left_key] = first_image
    if first_image.ndim not in {4, 5}:
        raise ValueError(
            "Expected batched camera tensor with shape (B, C, H, W) or (B, T, C, H, W). "
            f"Got {left_key} shape {tuple(first_image.shape)}."
        )

    batch_size = int(first_image.shape[0])
    if batch_size == 0:
        return batch

    draw = torch.rand(batch_size, device=first_image.device, generator=generator)
    left_mask = draw < config.left_prob
    front_mask = (draw >= config.left_prob) & (draw < config.left_prob + config.front_prob)
    right_mask = (
        (draw >= config.left_prob + config.front_prob)
        & (draw < config.left_prob + config.front_prob + config.right_prob)
    )

    masks = {
        left_key: left_mask,
        front_key: front_mask,
        right_key: right_mask,
    }
    for key, mask in masks.items():
        image = batch[key]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
            batch[key] = image
        if image.ndim not in {4, 5}:
            raise ValueError(
                "Expected batched camera tensor with shape (B, C, H, W) or (B, T, C, H, W). "
                f"Got {key} shape {tuple(image.shape)}."
            )
        if int(image.shape[0]) != batch_size:
            raise ValueError(
                f"Camera batch size mismatch: {key} has batch size {int(image.shape[0])}, "
                f"expected {batch_size}."
            )
        if mask.any():
            masked_image = image.clone()
            masked_image[mask] = config.fill
            batch[key] = masked_image

    return batch


def make_resize_pad_transform(config: ResizePadConfig) -> ResizePadSquare | None:
    if not config.enabled:
        return None
    return ResizePadSquare(size=config.size, fill=config.fill)


def resize_pad_feature_shape(shape: tuple[int, ...] | list[int], size: int) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected image feature shape (C, H, W). Got {shape}.")
    return (int(shape[0]), int(size), int(size))


def apply_resize_pad_to_feature_specs(feature_specs: dict[str, dict], config: ResizePadConfig) -> None:
    if not config.enabled:
        return
    for key, feature in feature_specs.items():
        if key.startswith("observation.images.") and feature.get("dtype") in {"image", "video"}:
            feature["shape"] = list(resize_pad_feature_shape(feature["shape"], config.size))


def infer_square_resize_pad_size_from_policy_features(policy_image_features: dict) -> int | None:
    sizes: set[int] = set()
    for feature in policy_image_features.values():
        shape = getattr(feature, "shape", None)
        if shape is None or len(shape) != 3:
            return None
        if int(shape[1]) != int(shape[2]):
            return None
        sizes.add(int(shape[1]))
    if len(sizes) != 1:
        return None
    return next(iter(sizes))
