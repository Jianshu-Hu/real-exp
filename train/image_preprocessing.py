from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.transforms import functional as F


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
