import socket
import time
from types import SimpleNamespace

import numpy as np
import zmq

from deploy.deploy_lerobot_policy import CameraBundleCache, resolve_server_local_observation


class FakeTimedObservation:
    def __init__(self, sequence: int | None = 4) -> None:
        self._observation = {"state_0": 0.1, "task": "test"}
        self.deployment_debug = {}
        if sequence is not None:
            self.deployment_debug["camera_bundle_sequence"] = sequence

    def get_observation(self):
        return self._observation


class FakeCache:
    def __init__(self, bundle):
        self.bundle = bundle

    def get(self, sequence):
        return self.bundle if sequence == 4 else None


POLICY_FEATURES = {
    "observation.images.cam_left": SimpleNamespace(shape=(3, 2, 3)),
    "observation.images.cam_right": SimpleNamespace(shape=(3, 2, 3)),
}
LIVE_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state_0"]},
    "observation.images.cam_left": {"dtype": "image", "shape": (2, 3, 3)},
    "observation.images.cam_right": {"dtype": "image", "shape": (2, 3, 3)},
}


def make_bundle(now_s: float | None = None):
    now_s = time.time() if now_s is None else now_s
    oldest_s = now_s - 0.02
    newest_s = now_s - 0.01
    return {
        "camera_bundle_sequence": 4,
        "bridge_publish_s": now_s,
        "robot_state_stamp_s": now_s - 0.015,
        "camera_sync": {
            "bundle_sequence": 4,
            "bundle_ready": True,
            "reference_stamp_s": oldest_s,
            "max_skew_s": newest_s - oldest_s,
        },
        "cameras": {
            "cam_left": {
                "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "shape": [2, 3, 3],
                "stamp_s": oldest_s,
            },
            "cam_right": {
                "rgb": np.ones((2, 3, 3), dtype=np.uint8),
                "shape": [2, 3, 3],
                "stamp_s": newest_s,
            },
        },
    }


def resolve(observation, bundle):
    return resolve_server_local_observation(
        observation,
        POLICY_FEATURES,
        LIVE_FEATURES,
        FakeCache(bundle),
        max_observation_age_s=0.25,
        max_camera_skew_s=0.067,
        rename_map=None,
    )


def test_resolves_exact_fresh_bundle_and_injects_all_policy_cameras():
    observation = FakeTimedObservation()
    bundle = make_bundle()
    observation.deployment_debug["robot_state_stamp_s"] = bundle["robot_state_stamp_s"]
    result = resolve(observation, bundle)
    assert result is observation
    np.testing.assert_array_equal(result.get_observation()["cam_right"], np.ones((2, 3, 3)))
    assert result.deployment_debug["camera_computed_skew_s_at_server"] <= 0.067
    assert result.deployment_debug["state_camera_skew_s_at_server"] <= 0.067


def test_rejects_missing_bundle_reference_when_rgb_is_absent():
    observation = FakeTimedObservation(sequence=None)
    try:
        resolve(observation, make_bundle())
    except RuntimeError as exc:
        assert "no camera_bundle_sequence" in str(exc)
    else:
        raise AssertionError("missing bundle reference was accepted")


def test_rejects_incomplete_stale_and_unsynchronized_bundles():
    incomplete = make_bundle()
    incomplete["cameras"].pop("cam_right")
    stale = make_bundle(time.time() - 1.0)
    skewed = make_bundle()
    skewed["cameras"]["cam_right"]["stamp_s"] = time.time() + 0.2

    for bundle, expected_message in (
        (incomplete, "missing policy cameras"),
        (stale, "outside the allowed window"),
        (skewed, "computed skew"),
    ):
        try:
            resolve(FakeTimedObservation(), bundle)
        except RuntimeError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"invalid bundle was accepted: {expected_message}")


def test_state_only_policy_bypasses_camera_cache_resolution():
    observation = FakeTimedObservation(sequence=None)
    result = resolve_server_local_observation(
        observation,
        policy_image_features={},
        lerobot_features={
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["state_0"],
            }
        },
        camera_bundle_cache=None,
        max_observation_age_s=0.25,
        max_camera_skew_s=0.067,
    )
    assert result is observation


def test_rename_map_resolves_only_live_cameras_used_by_policy():
    observation = FakeTimedObservation()
    bundle = make_bundle()
    bundle["cameras"].pop("cam_right")
    observation.deployment_debug["robot_state_stamp_s"] = bundle["robot_state_stamp_s"]
    result = resolve_server_local_observation(
        observation,
        policy_image_features={
            "observation.images.policy_left": SimpleNamespace(shape=(3, 2, 3))
        },
        lerobot_features=LIVE_FEATURES,
        camera_bundle_cache=FakeCache(bundle),
        max_observation_age_s=0.25,
        max_camera_skew_s=0.067,
        rename_map={"observation.images.cam_left": "observation.images.policy_left"},
    )
    assert "cam_left" in result.get_observation()
    assert "cam_right" not in result.get_observation()


def _unused_tcp_address() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def test_loopback_cache_exact_lookup_and_bounded_eviction():
    address = _unused_tcp_address()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(address)
    cache = CameraBundleCache(address, max_entries=2)
    cache.start()
    try:
        # PUB/SUB needs a short subscription handshake. Repetition also makes this test robust
        # against the documented initial slow-joiner drop.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and cache.get(3) is None:
            for sequence in (1, 2, 3):
                publisher.send_pyobj(
                    {"camera_bundle_sequence": sequence, "cameras": {"cam": {"rgb": 1}}}
                )
            time.sleep(0.02)
        assert cache.get(3) is not None
        assert cache.get(2) is not None
        assert cache.get(1) is None
        assert cache.get(99) is None
    finally:
        cache.close()
        publisher.close(0)
        context.term()
