"""JSON request/reply contract between perception server and control client."""

import math
import time
import uuid

FORMAT = "softhand_grasp.v1"


def request_message(description, instance_index=None):
    return {"format": FORMAT, "request_id": str(uuid.uuid4()),
            "created_unix_s": time.time(), "action": "infer_grasp",
            "side": "right", "description": description, "instance_index": instance_index}


def check_age(value, max_age_s, name, now=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite timestamp")
    age = (time.time() if now is None else now) - value
    if age < -5 or age > max_age_s:
        raise ValueError(f"{name} is stale or clocks are not synchronized (age {age:.1f} s)")


def validate_request(request):
    if not isinstance(request, dict) or request.get("format") != FORMAT:
        raise ValueError("Invalid grasp request format")
    if request.get("action") != "infer_grasp" or request.get("side") != "right":
        raise ValueError("Only right-arm infer_grasp requests are supported")
    if not isinstance(request.get("request_id"), str) or not request["request_id"]:
        raise ValueError("request_id is required")
    if not isinstance(request.get("description"), str) or not request["description"].strip():
        raise ValueError("Object description is required")
    rank = request.get("instance_index")
    if rank is not None and (type(rank) is not int or rank < 0):
        raise ValueError("instance_index must be a nonnegative integer")
    check_age(request.get("created_unix_s"), 30, "request timestamp")
    return request


def validate_response(response, request, *, execute=False, max_age_s=300):
    from softhand_grasp.geometry import validate_pose

    if not isinstance(response, dict) or response.get("format") != FORMAT:
        raise ValueError("Invalid grasp response format")
    if response.get("request_id") != request["request_id"]:
        raise ValueError("Grasp response request_id mismatch")
    if response.get("ok") is not True:
        raise RuntimeError(f"Server inference failed: {response.get('error', 'unknown error')}")
    if response.get("side") != "right" or response.get("pose_frame") != "right_base":
        raise ValueError("Server poses must be in the right-arm base frame")
    if type(response.get("live_capture")) is not bool:
        raise ValueError("Server must identify whether the capture is live")
    if execute and not response["live_capture"]:
        raise ValueError("Cannot execute a plan from a replay capture")
    check_age(response.get("capture_started_unix_s"), max_age_s, "capture timestamp")
    check_age(response.get("completed_unix_s"), max_age_s, "response timestamp")
    if response["completed_unix_s"] < response["capture_started_unix_s"]:
        raise ValueError("Response completion precedes capture")
    if response["capture_started_unix_s"] < request["created_unix_s"] - 5:
        raise ValueError("Capture precedes this request")
    plan = response.get("plan")
    if not isinstance(plan, dict) or plan.get("description") != request["description"]:
        raise ValueError("Server plan description mismatch")
    for key in ("grasp_pose_xyzrpy", "lift_pose_xyzrpy"):
        validate_pose(plan.get(key))
    return plan
