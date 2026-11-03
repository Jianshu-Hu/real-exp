"""SAM3 text segmentation of a saved, aligned front-D435i capture."""

import argparse
import json
from pathlib import Path

import numpy as np

from softhand_grasp.geometry import DEFAULT_CONFIG, load_config, surface_point, target_poses


def segment(rgb, description, checkpoint, device, threshold):
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Native SAM3 requires an available CUDA device")
    # Native SAM3 creates positional caches on the current CUDA device.
    with torch.cuda.device(device), torch.inference_mode():
        model = build_sam3_image_model(checkpoint_path=str(checkpoint),
                                      load_from_HF=False, device="cpu").to(device).eval()
        processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        # Native SAM3 attention uses bfloat16; keep its surrounding layers in
        # the same autocast context rather than mixing bf16 activations/fp32 weights.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(rgb))
            result = processor.set_text_prompt(prompt=description, state=state)
        return (result["masks"].detach().cpu().numpy(),
                result["scores"].detach().float().cpu().numpy())


def choose_segment(masks, scores, depth_m, intrinsics, config, instance_index=None):
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim != 3 or masks.shape[1:] != depth_m.shape or len(masks) != len(scores):
        raise ValueError("SAM3 returned incompatible mask/score shapes")
    # An instance is one object; never union masks belonging to separate objects.
    candidates = []
    for index in np.argsort(-scores, kind="stable"):
        if not np.isfinite(scores[index]) or scores[index] < config["score_threshold"]:
            continue
        mask = masks[index] > 0.5
        try:
            point, count = surface_point(mask, depth_m, intrinsics, config)
        except ValueError:
            continue
        candidates.append((int(index), mask, point, count))
    if not candidates:
        raise ValueError("No text-matched SAM3 instance has sufficient valid depth")
    rank = 0 if instance_index is None else instance_index
    if not 0 <= rank < len(candidates):
        raise ValueError(f"instance-index must be in [0, {len(candidates) - 1}]")
    index, mask, point, count = candidates[rank]
    return mask, point, {"sam3_index": index, "instance_rank": rank,
                         "score": float(scores[index]), "depth_inliers": count,
                         "valid_instance_count": len(candidates)}


def infer(capture_dir, output_dir, description, config, checkpoint, device="cuda", instance_index=None):
    from PIL import Image

    capture_dir, output_dir = Path(capture_dir), Path(output_dir)
    metadata = json.loads((capture_dir / "metadata.json").read_text())
    intrinsics = metadata["intrinsics"]
    if str(metadata["camera_serial"]) != config["camera_serial"]:
        raise ValueError("Captured camera serial does not match the calibrated front camera")
    if (intrinsics["width"], intrinsics["height"]) != (640, 480):
        raise ValueError("Front-camera calibration requires the 640x480 color profile")
    # grasp.realsense_capture stores BGR despite its historical rgb.npy name.
    bgr = np.load(capture_dir / "rgb.npy", allow_pickle=False)
    depth_raw = np.load(capture_dir / "depth.npy", allow_pickle=False)
    if bgr.shape != (480, 640, 3) or bgr.dtype != np.uint8 or depth_raw.shape != bgr.shape[:2]:
        raise ValueError("Capture must contain 640x480 BGR uint8 and aligned depth")
    scale = float(metadata["depth_scale_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale_m must be positive and finite")
    rgb = bgr[..., ::-1].copy()
    masks, scores = segment(rgb, description, checkpoint, device, config["score_threshold"])
    mask, point, selection = choose_segment(masks, scores, depth_raw * scale, intrinsics, config, instance_index)
    plan = {"description": description, "camera_serial": config["camera_serial"],
            "capture_dir": str(capture_dir.resolve()),
            "checkpoint": str(Path(checkpoint).resolve()), "device": device,
            "selection": selection, **target_poses(point, config)}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "mask.npy", mask)
    overlay = rgb.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(rgb).save(output_dir / "rgb.png")
    Image.fromarray(overlay).save(output_dir / "selected_mask.png")
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--object", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instance-index", type=int)
    args = parser.parse_args()
    if not args.object.strip():
        parser.error("--object must be a nonempty description")
    infer(args.capture_dir, args.output_dir, args.object, load_config(args.config),
          args.checkpoint, args.device, args.instance_index)


if __name__ == "__main__":
    main()
