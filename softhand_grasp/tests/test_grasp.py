import json

import numpy as np
import pytest

from softhand_grasp import perception, client as run
from softhand_grasp.geometry import load_config, surface_point, target_poses


@pytest.fixture
def config():
    return {**load_config(), **run.load_config()}


def test_surface_deprojection_rejects_holes_and_outliers(config):
    config.update(mask_erode_pixels=1, min_depth_pixels=10)
    mask = np.ones((9, 9), dtype=bool)
    depth = np.full((9, 9), 0.8)
    depth[0, :] = 1.5  # Boundary is removed by erosion.
    depth[1, 1], depth[1, 2], depth[1, 3] = 0, np.nan, 1.8
    intrinsics = dict(fx=100, fy=100, cx=4, cy=4, coeffs=[0] * 5)
    point, count = surface_point(mask, depth, intrinsics, config)
    np.testing.assert_allclose(point, [0, 0, 0.8])
    assert count == 46
    with pytest.raises(ValueError, match="too few"):
        surface_point(mask, np.zeros_like(depth), intrinsics, config)
    with pytest.raises(ValueError, match="distortion"):
        surface_point(mask, depth, dict(intrinsics, coeffs=[0.1] * 5), config)


def test_transform_direction_tool_offset_and_world_lift(config):
    # Independent expected geometry: +90 deg camera rotation about Z, with translation.
    config["camera_T_right_base"] = [[0, -1, 0, 0.2], [1, 0, 0, 0.1],
                                     [0, 0, 1, 0.3], [0, 0, 0, 1]]
    config["world_T_camera"] = np.eye(4).tolist()
    config["grasp_rpy"] = [0, np.pi / 2, 0]
    config["ee_to_grasp_point_m"] = [0, 0, 0.1]
    poses = target_poses([0.1, 0.5, 0.8], config)
    np.testing.assert_allclose(poses["object_point_base_m"], [0.4, 0.1, 0.5])
    np.testing.assert_allclose(poses["grasp_pose_xyzrpy"][:3], [0.3, 0.1, 0.5])
    np.testing.assert_allclose(poses["lift_pose_xyzrpy"][:3], [0.3, 0.1, 0.6])
    config["lift_frame"] = "ee"
    poses = target_poses([0.1, 0.5, 0.8], config)
    np.testing.assert_allclose(poses["lift_pose_xyzrpy"][:3], [0.4, 0.1, 0.5])


def test_installed_calibration_lifts_exactly_world_vertical(config):
    result = target_poses([0, 0, 0.65], config)
    delta_base = np.subtract(result["lift_pose_xyzrpy"][:3], result["grasp_pose_xyzrpy"][:3])
    world_T_base = np.array(config["world_T_camera"]) @ np.array(config["camera_T_right_base"])
    np.testing.assert_allclose(world_T_base[:3, :3] @ delta_base, [0, 0, 0.1], atol=1e-9)
    assert result["lift_pose_xyzrpy"][3:] == result["grasp_pose_xyzrpy"][3:]


def test_instance_selection_does_not_average_objects(config):
    config.update(mask_erode_pixels=0, min_depth_pixels=4)
    masks = np.zeros((3, 1, 10, 10), dtype=bool)
    masks[0, 0, :3, :3] = True
    masks[1, 0, 3:6, 3:6] = True
    masks[2, 0, 7:, 7:] = True
    depth = np.ones((10, 10)) * 0.5
    depth[:3, :3] = 0  # Highest-score object has no depth.
    intrinsics = dict(fx=100, fy=100, cx=4, cy=4)
    mask, point, info = perception.choose_segment(masks, [0.99, 0.9, 0.8], depth, intrinsics, config)
    assert info["sam3_index"] == 1 and info["valid_instance_count"] == 2
    assert mask.sum() == 9
    np.testing.assert_allclose(point, [0, 0, 0.5])
    _, _, info = perception.choose_segment(masks, [0.99, 0.9, 0.8], depth, intrinsics, config, 1)
    assert info["sam3_index"] == 2
    with pytest.raises(ValueError, match="No text-matched"):
        perception.choose_segment(masks, [0.1] * 3, depth, intrinsics, config)


class FakeArm:
    def __init__(self, events, fail_move=None):
        self.events, self.count, self.fail_move = events, 0, fail_move

    def move_to(self, pose):
        self.count += 1
        self.events.append(f"move{self.count}")
        if self.count == self.fail_move:
            raise RuntimeError("arrival failed")


class FakePublisher:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail

    def __enter__(self):
        self.events.append("open")
        return self

    def publish_all(self, values):
        assert values == [5] * 20
        self.events.append("publish")
        if self.fail:
            raise RuntimeError("pressure write failed")

    def __exit__(self, *_):
        self.events.append("all_off")


def test_full_sequence_and_dwell(config):
    events, now, sent_at = [], [0.0], []
    arm = FakeArm(events)
    publisher = FakePublisher(events)

    def perceive():
        events.append("capture_sam3")
        return target_poses([0, 0, 0.65], config)

    def sleep(dt):
        now[0] += dt

    def pressurize(p, cfg):
        run.publish_for(p, cfg, clock=lambda: now[0], sleep=sleep)
        sent_at.append(now[0])

    run.run_sequence(config, arm, perceive, execute=True,
                     make_publisher=lambda cfg: publisher,
                     pressurize=pressurize, hold=lambda: events.append("hold"))
    assert events[:4] == ["move1", "capture_sam3", "move2", "open"]
    assert events[-3:] == ["move3", "hold", "all_off"]
    assert events.count("publish") >= 40 and sent_at[0] == pytest.approx(2.0)


@pytest.mark.parametrize("failure", ["pregrasp", "perception", "approach", "pressure", "lift"])
def test_failure_aborts_subsequent_phases(config, failure):
    events = []
    arm = FakeArm(events, {"pregrasp": 1, "approach": 2, "lift": 3}.get(failure))

    def perceive():
        events.append("perception")
        if failure == "perception":
            raise RuntimeError("no object")
        return target_poses([0, 0, 0.65], config)

    with pytest.raises(RuntimeError):
        run.run_sequence(config, arm, perceive, execute=True,
                         make_publisher=lambda cfg: FakePublisher(events, failure == "pressure"),
                         pressurize=lambda p, cfg: p.publish_all(cfg["channel_voltages"]),
                         hold=lambda: events.append("hold"))
    if failure in ("pregrasp", "perception", "approach"):
        assert "publish" not in events and "open" not in events
    if failure != "lift":
        assert "move3" not in events
    if failure in ("pressure", "lift"):
        assert events[-1] == "all_off"
    assert "hold" not in events


def test_preview_never_opens_pneumatics(config):
    def forbidden(*args):
        pytest.fail("Preview attempted hardware access")
    events = []
    run.run_sequence(config, FakeArm(events), lambda: target_poses([0, 0, 0.65], config),
                     make_publisher=forbidden, hold=forbidden, pressurize=forbidden)
    assert events == ["move1", "move2", "move3"]


@pytest.mark.parametrize("field,value", [("channel_voltages", [6] * 20),
    ("channel_voltages", [5] * 16), ("close_duration_s", float("nan")),
    ("lift_frame", "invalid"), ("camera_T_right_base", np.zeros((4, 4)).tolist()),
    ("pregrasp_pose_xyzrpy", [2, 0, 0, 0, 0, 0])])
def test_config_rejects_invalid_commands(tmp_path, config, field, value):
    config[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        (run.load_config if field in run.load_config() else load_config)(path)


def test_inference_artifacts_bgr_and_depth_scale(tmp_path, config, monkeypatch):
    capture, output = tmp_path / "capture", tmp_path / "output"
    capture.mkdir()
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr[:] = [10, 20, 200]
    np.save(capture / "rgb.npy", bgr)
    np.save(capture / "depth.npy", np.full((480, 640), 650, dtype=np.uint16))
    metadata = dict(camera_serial=config["camera_serial"], depth_scale_m=0.001,
                    intrinsics=dict(width=640, height=480, fx=600, fy=600, cx=320, cy=240, coeffs=[0]*5))
    (capture / "metadata.json").write_text(json.dumps(metadata))

    def segment(rgb, description, *_):
        np.testing.assert_array_equal(rgb[0, 0], [200, 20, 10])
        assert description == "red cup"
        masks = np.zeros((1, 1, 480, 640), bool)
        masks[0, 0, 225:256, 305:336] = True
        return masks, np.array([0.9])

    monkeypatch.setattr(perception, "segment", segment)
    result = perception.infer(capture, output, "red cup", config, "unused.pt")
    np.testing.assert_allclose(result["object_point_camera_m"], [0, 0, 0.65])
    assert {"rgb.png", "selected_mask.png", "mask.npy", "plan.json"} <= {p.name for p in output.iterdir()}
    metadata["camera_serial"] = "wrong_camera"
    (capture / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="serial"):
        perception.infer(capture, output, "red cup", config, "unused.pt")


def test_client_needs_only_control_configuration():
    args, config = run.checked_args(["--object", "cup", "--validate-only"])
    assert args.object == "cup"
    assert "camera_T_right_base" not in config
    assert not hasattr(args, "checkpoint")
    assert not hasattr(args, "camera_python")


def test_real_publisher_pads_channels_and_stays_open_through_lift(config, monkeypatch):
    # Use the actual submodule Publisher with only its physical serial driver replaced.
    monkeypatch.syspath_prepend(str(run.ROOT / "PneumaticSystem"))
    from Service.publisher import Publisher
    events = []

    class Driver:
        connected = False

        def connect(self):
            self.connected = True
            events.append("connect")

        def set_all_outputs_v(self, values):
            assert self.connected
            assert values == (5.0,) * 20 + (0.0,) * 4
            events.append("write24")

        def disconnect(self, all_off):
            assert all_off
            self.connected = False
            events.append("all_off")

    driver = Driver()
    publisher = Publisher(driver=driver)

    class Arm(FakeArm):
        def move_to(self, pose):
            if self.count == 2:
                assert driver.connected
                assert events[-1] == "write24"
            super().move_to(pose)

    run.run_sequence(config, Arm(events), lambda: target_poses([0, 0, 0.65], config),
                     execute=True, make_publisher=lambda cfg: publisher,
                     pressurize=lambda p, cfg: p.publish_all(cfg["channel_voltages"]),
                     hold=lambda: events.append("hold"))
    assert events == ["move1", "move2", "connect", "write24", "move3", "hold", "all_off"]


@pytest.mark.parametrize("reached", [True, False])
def test_arm_adapter_uses_right_planner_and_measured_arrival(monkeypatch, reached):
    from types import SimpleNamespace
    from softhand_grasp.arm import RightArm, move

    calls = []
    arm = RightArm()
    arm.ros = object()
    arm.node = SimpleNamespace(ee_pose={}, arm_q={}, flange_to_ee={})
    arm.model, arm.frame_id = object(), 7
    arm.validity, arm.joint_constraint, arm.constraints, arm.action = [object() for _ in range(4)]
    q = np.array([0, 0, 0, -1, 0, 1, 0.])
    trajectory = object()

    def fresh(ros, node):
        assert ros is arm.ros and node is arm.node
        assert node.ee_pose["right"] is None
        node.ee_pose["right"] = np.eye(4)
        node.arm_q["right"] = q.copy()
        node.flange_to_ee["right"] = np.eye(4)
        calls.append("fresh")

    def plan(*args):
        assert args[2] == "right"
        np.testing.assert_allclose(args[4][:3, 3], [0.4, 0.2, 0.5])
        calls.append("plan")
        return trajectory, 0.2, None

    def audit(*args):
        assert args[0] == "right" and args[1] is trajectory
        calls.append("audit")

    def endpoint(*args):
        calls.append("endpoint")
        return q

    def execute(*args):
        assert args[2] == "right" and args[3] is trajectory and args[4] is arm.action
        calls.append("execute")
        return True

    def measure(*args):
        assert args[2] == "right"
        calls.append("measure")
        return reached, 0.0 if reached else 0.05, 0.0

    monkeypatch.setattr(move, "wait_for_robot_state", fresh)
    monkeypatch.setattr(move, "solve_fr3_ik", lambda *a, **kw: SimpleNamespace(q=q))
    monkeypatch.setattr(move, "plan_deterministic_cartesian_trajectory", plan)
    monkeypatch.setattr(move, "audit_trajectory", audit)
    monkeypatch.setattr(move, "print_trajectory_summary", lambda *a: None)
    monkeypatch.setattr(move, "verify_planned_endpoint", endpoint)
    monkeypatch.setattr(move, "execute_joint_trajectory", execute)
    monkeypatch.setattr(move, "measure_final_ee_pose", measure)
    if reached:
        arm.move_to([0.4, 0.2, 0.5, 0, 0, 0])
    else:
        with pytest.raises(RuntimeError, match="arrival failed"):
            arm.move_to([0.4, 0.2, 0.5, 0, 0, 0])
    assert calls == ["fresh", "plan", "audit", "endpoint", "execute", "measure"]
