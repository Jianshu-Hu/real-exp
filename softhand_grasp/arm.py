"""Right-arm adapter for the existing ROS/MoveIt Cartesian motion utilities."""

from types import SimpleNamespace

import numpy as np

from data_collection import move_to_target_ee as move
from softhand_grasp.geometry import validate_pose


class RightArm:
    def __enter__(self):
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from franka_msgs.msg import FrankaRobotState
        from moveit_msgs.msg import Constraints, JointConstraint
        from moveit_msgs.srv import GetStateValidity
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        self.ros = rclpy
        self.action = FollowJointTrajectory
        self.validity = GetStateValidity
        self.joint_constraint = JointConstraint
        self.constraints = Constraints
        rclpy.init()
        self.node = None
        try:
            self.node = move.build_move_node_class(Node, JointState, FrankaRobotState)(
                SimpleNamespace(arm_mode="right"))
            move.wait_for_robot_state(self.ros, self.node)
            self.model, self.frame_id = move.build_fr3_model()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def move_to(self, pose):
        """Return only after the measured Cartesian arrival check succeeds."""
        target = move.pose_vector_to_matrix(validate_pose(pose))
        node, ros = self.node, self.ros
        # Capture new state after potentially long perception/pressure phases.
        node.ee_pose["right"] = node.arm_q["right"] = node.flange_to_ee["right"] = None
        move.wait_for_robot_state(ros, node)
        start_pose = node.ee_pose["right"].copy()
        start_q = node.arm_q["right"].copy()
        flange = node.flange_to_ee["right"].copy()
        reference = move.solve_fr3_ik(
            start_q, target, flange, self.model, self.frame_id,
            try_alternative_seeds=False,
            position_lower_rad=np.maximum(move.ARM_POSITION_LOWER_RAD,
                                          start_q - move.CARTESIAN_REFERENCE_MAX_JOINT_CHANGE_RAD),
            position_upper_rad=np.minimum(move.ARM_POSITION_UPPER_RAD,
                                          start_q + move.CARTESIAN_REFERENCE_MAX_JOINT_CHANGE_RAD))
        trajectory, duration, _ = move.plan_deterministic_cartesian_trajectory(
            ros, node, "right", start_pose, target, flange, start_q, reference.q,
            self.validity, self.joint_constraint, self.constraints)
        audit = move.audit_trajectory("right", trajectory, start_pose, target, flange, reference.q)
        move.print_trajectory_summary("right", trajectory, duration, audit)
        final_q = move.verify_planned_endpoint("right", trajectory, target, flange)
        move.execute_joint_trajectory(ros, node, "right", trajectory, self.action)
        reached, position_error, rotation_error = move.measure_final_ee_pose(
            ros, node, "right", target, final_q)
        if not reached:
            raise RuntimeError(f"Right-arm arrival failed: {position_error:.4f} m, {rotation_error:.4f} rad")

    def __exit__(self, *_):
        if self.node is not None:
            self.node.destroy_node()
        if self.ros.ok():
            self.ros.shutdown()


class PreviewArm:
    """Pose preview only: does not claim IK, collision, or arrival validation."""

    def __enter__(self):
        return self

    def move_to(self, pose):
        print("Preview right EE xyzrpy:", validate_pose(pose).tolist(), flush=True)

    def __exit__(self, *_):
        pass
