"""Offline regression coverage for grasp IK and bounded transit fallback."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from data_collection import move_to_target_ee as move
from utils.fr3_kinematics import Fr3ForwardKinematics


GRASP = [0.6016670626115803, -0.13054184735946128, 0.14918996535885187,
         -2.0214924347110745, 0.6999574280463707, -2.2677496905437993]
STARTS = [
    [-0.887510, 0.113785, 0.690168, -1.736973, -1.608149, 3.147324, -1.516071],
    [-0.577932, 0.129896, 0.424003, -1.701765, -1.543360, 3.021550, -1.565335],
]


@pytest.mark.parametrize("joints", STARTS)
def test_grasp_escapes_failed_local_ik_without_relaxing_pose_or_joint_limits(joints):
    fk = Fr3ForwardKinematics(backend="numpy")
    q = np.asarray(joints)
    tool = move.pose_vector_to_matrix([0, 0, 0.1034, 0, 0, -np.pi / 4])
    target = move.pose_vector_to_matrix(GRASP)
    bounds = dict(position_lower_rad=np.maximum(move.ARM_POSITION_LOWER_RAD, q - 2.6),
                  position_upper_rad=np.minimum(move.ARM_POSITION_UPPER_RAD, q + 2.6))
    with pytest.raises(move.IkSolveError, match="verified pose tolerance"):
        move.solve_fr3_ik(q, target, tool, kinematics=fk, try_alternative_seeds=False, **bounds)
    result = move.solve_fr3_ik(q, target, tool, kinematics=fk, **bounds)
    position_error, rotation_error = move.pose_error(fk.end_effector_pose(result.q, tool), target)
    assert position_error < move.MOVEIT_POSITION_TOLERANCE_M
    assert rotation_error < move.IK_ORIENTATION_TOLERANCE_RAD
    assert np.all(result.q >= bounds["position_lower_rad"])
    assert np.all(result.q <= bounds["position_upper_rad"])
    assert np.all(result.q >= move.ARM_POSITION_LOWER_RAD + 0.05)
    assert np.all(result.q <= move.ARM_POSITION_UPPER_RAD - 0.05)


def test_reachable_local_ik_skips_alternative_search(monkeypatch):
    import scipy.optimize

    fk = Fr3ForwardKinematics(backend="numpy")
    q = np.array(STARTS[1])
    calls = []
    original = scipy.optimize.least_squares

    def solve(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "least_squares", solve)
    move.solve_fr3_ik(q, fk.flange_pose(q), np.eye(4), kinematics=fk)
    assert len(calls) == 1


def test_transit_goal_uses_explicit_joints_and_bounded_path_constraints():
    class Constraints:
        def __init__(self):
            self.joint_constraints = []
            self.position_constraints = []
            self.orientation_constraints = []

    class Position:
        def __init__(self):
            self.header = SimpleNamespace()
            self.constraint_region = SimpleNamespace()

    class Orientation:
        def __init__(self):
            self.header = SimpleNamespace()

    class Primitive:
        SPHERE = 2

    def goal():
        return SimpleNamespace(request=SimpleNamespace(
            start_state=SimpleNamespace(joint_state=SimpleNamespace())),
            planning_options=SimpleNamespace())

    types = dict(MoveGroup=SimpleNamespace(Goal=goal), Constraints=Constraints,
                 PositionConstraint=Position, OrientationConstraint=Orientation,
                 JointConstraint=SimpleNamespace, SolidPrimitive=Primitive,
                 Pose=lambda: SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace()))
    start = np.asarray(STARTS[1])
    end = start + 0.1
    result = move.build_move_group_goal("right", np.eye(4), np.eye(4), types,
                                        current_q=start, target_q=end)
    assert result.planning_options.plan_only
    assert result.request.start_state.joint_state.position == start.tolist()
    joint_goal = result.request.goal_constraints[0]
    assert not joint_goal.position_constraints
    assert len(joint_goal.joint_constraints) == 7
    lower, upper = move.cartesian_joint_corridor(start, end)
    for i, (joint, bound) in enumerate(zip(joint_goal.joint_constraints,
                                         result.request.path_constraints.joint_constraints)):
        assert joint.joint_name == bound.joint_name == f"right_fr3_joint{i+1}"
        assert joint.position == end[i]
        assert bound.position - bound.tolerance_below == pytest.approx(lower[i])
        assert bound.position + bound.tolerance_above == pytest.approx(upper[i])


@pytest.mark.parametrize("failure_stage", ["reference", "waypoint"])
def test_ik_failure_can_use_joint_transit_only_when_allowed(monkeypatch, failure_stage):
    q = np.asarray(STARTS[1])
    target_q = q + 0.1
    trajectory = object()
    events = []

    def solve(*args, **kwargs):
        events.append(("ik", kwargs["try_alternative_seeds"]))
        if failure_stage == "reference" and not kwargs["try_alternative_seeds"]:
            raise move.IkSolveError("local minimum")
        return SimpleNamespace(q=target_q)

    def cartesian(*args):
        events.append("cartesian")
        raise move.IkSolveError("waypoint infeasible")

    def transit(*args, **kwargs):
        events.append("ompl")
        np.testing.assert_allclose(kwargs["target_q"], target_q)
        return trajectory, 0.1

    def prepare(*args):
        events.append("check")

    monkeypatch.setattr(move, "solve_fr3_ik", solve)
    monkeypatch.setattr(move, "plan_deterministic_cartesian_trajectory", cartesian)
    monkeypatch.setattr(move, "plan_moveit_trajectory", transit)
    monkeypatch.setattr(move, "prepare_joint_transit", prepare)
    args = (object(), object(), "right", np.eye(4), np.eye(4), np.eye(4), q,
            dict(GetStateValidity=object, JointConstraint=object, Constraints=object))
    with pytest.raises(move.IkSolveError):
        move.plan_fr3_trajectory(*args, model=object(), frame_id=7)
    assert "ompl" not in events
    result = move.plan_fr3_trajectory(*args, model=object(), frame_id=7,
                                    allow_joint_reconfiguration=True)
    assert result[0] is trajectory and result[3] is False
    assert events[-3:] == [("ik", True), "ompl", "check"]


def test_collision_failure_never_enables_transit_fallback(monkeypatch):
    q = np.asarray(STARTS[1])
    monkeypatch.setattr(move, "solve_fr3_ik", lambda *a, **kw: SimpleNamespace(q=q))

    def collision(*args):
        raise RuntimeError("invalid in planning scene")

    def forbidden(*args, **kwargs):
        pytest.fail("Collision failure enabled fallback")

    monkeypatch.setattr(move, "plan_deterministic_cartesian_trajectory", collision)
    monkeypatch.setattr(move, "plan_moveit_trajectory", forbidden)
    with pytest.raises(RuntimeError, match="invalid in planning scene"):
        move.plan_fr3_trajectory(
            object(), object(), "right", np.eye(4), np.eye(4), np.eye(4), q,
            dict(GetStateValidity=object, JointConstraint=object, Constraints=object),
            model=object(), frame_id=7, allow_joint_reconfiguration=True,
        )


@pytest.fixture
def point_type(monkeypatch):
    class Point:
        def __init__(self, positions):
            self.positions = positions
            self.time_from_start = SimpleNamespace(sec=0, nanosec=0)

    module = SimpleNamespace(JointTrajectoryPoint=Point)
    monkeypatch.setitem(sys.modules, "trajectory_msgs", SimpleNamespace(msg=module))
    monkeypatch.setitem(sys.modules, "trajectory_msgs.msg", module)
    return Point


@pytest.mark.parametrize("failure", [None, "collision", "bounds", "start", "nonfinite"])
def test_joint_transit_checks_intermediate_states_before_retiming(monkeypatch, point_type, failure):
    q = np.asarray(STARTS[1])
    end = q + 0.4
    path = [q.copy(), end.copy()]
    if failure == "bounds":
        path.insert(1, end + 0.4)
    elif failure == "start":
        path[0] += 0.04
    elif failure == "nonfinite":
        path[1][0] = np.nan
    trajectory = SimpleNamespace(
        joint_names=[f"right_{name}" for name in move.JOINT_NAMES],
        points=[point_type(item.tolist()) for item in path],
    )
    checked, destroyed = [], []
    client = SimpleNamespace(wait_for_service=lambda **kw: True)
    node = SimpleNamespace(create_client=lambda *a: client,
                           destroy_client=lambda c: destroyed.append(c))

    def check(*args):
        checked.append(args[3])
        if failure == "collision" and len(checked) > 2:
            raise RuntimeError("collision")

    monkeypatch.setattr(move, "check_moveit_state_validity", check)
    types = dict(Constraints=object, GetStateValidity=object)
    if failure:
        with pytest.raises(RuntimeError):
            move.prepare_joint_transit(object(), node, "right", trajectory, q, end, types)
        if failure in ("collision", "bounds"):
            assert destroyed == [client]
    else:
        move.prepare_joint_transit(object(), node, "right", trajectory, q, end, types)
        checked = np.stack(checked)
        assert len(checked) >= 9
        assert np.max(np.abs(np.diff(checked, axis=0))) <= 0.05 + 1e-9
        assert destroyed == [client]
        assert move.trajectory_duration_s(trajectory) > 0
        np.testing.assert_allclose(checked[-1], end)


def test_joint_transit_audit_allows_curvature_but_still_rejects_joint_jumps(monkeypatch, point_type):
    fk = Fr3ForwardKinematics(backend="numpy")
    q = np.asarray(STARTS[1])
    middle = q + 0.1
    target = fk.flange_pose(q)
    trajectory = SimpleNamespace(
        joint_names=[f"right_{name}" for name in move.JOINT_NAMES],
        points=[point_type(item.tolist()) for item in (q, middle, q)],
    )
    monkeypatch.setattr(move, "build_fr3_model", lambda: (object(), 7))
    monkeypatch.setattr(move, "forward_end_effector_pose", lambda m, f, joints, tool: fk.end_effector_pose(joints, tool))
    with pytest.raises(RuntimeError, match="deviation"):
        move.audit_trajectory("right", trajectory, target, target, np.eye(4), q)
    move.audit_trajectory("right", trajectory, target, target, np.eye(4), q, require_cartesian=False)
    trajectory.points[1].positions = (q + 0.25).tolist()
    with pytest.raises(RuntimeError, match="joint step"):
        move.audit_trajectory("right", trajectory, target, target, np.eye(4), q, require_cartesian=False)
