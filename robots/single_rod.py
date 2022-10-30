import numpy as np
from collections import OrderedDict

from robosuite.controllers import controller_factory
from robosuite.robots.robot import Robot


class SingleRod(Robot):
    """
    soft rod: imitated by one joint and one link robot
    """
    def __init__(self,
                 robot_type,
                 idn,
                 initial_qpos,
                 initialization_noise=None,
                 control_freq=10
                 ):

        self.controller = None
        self.controller_config = None
        self.rod_top_id = None
        self.control_freq = control_freq

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
        )

    def _load_controller(self):
        self.controller_config["robot_name"] = self.name
        self.controller_config["sim"] = self.sim
        self.controller_config["joint_indexes"] = {
            "joints": self.joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes
                                              }
        self.controller_config["actuator_range"] = (np.array(-80), np.array(80))
        self.controller_config["policy_freq"] = 20
        self.controller_config["ndim"] = len(self.robot_joints)

        self.controller = controller_factory(self.controller_config["type"], self.controller_config)

    def load_model(self):
        super().load_model()

    def reset(self, deterministic=False):
        super().reset(deterministic)
        # Update base pos / ori references in controller

        self.controller.update_base_pose(self.base_pos, self.base_ori)

    def setup_references(self):
        super().setup_references()
        self.rod_top_id = self.sim.model.site_name2id(self.robot_model.naming_prefix + "rodtop")

    def control(self, action, policy_step=False):
        # Update the controller goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(action)

        # Now run the controller for a step
        torques = self.controller.run_controller()
        # Clip the torques
        low, high = self.torque_limits
        torques = np.clip(torques, low, high)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_torq_actuator_indexes] = torques

    def visualize_gripper(self):
        raise Exception("No need to call this function!")

    def get_observations(self, di: OrderedDict):
        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robot_model.naming_prefix

        # proprioceptive features
        di[pf + "joint_pos"] = np.array([self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])
        di[pf + "joint_vel"] = np.array([self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])

        robot_states = [
            np.sin(di[pf + "joint_pos"]),
            np.cos(di[pf + "joint_pos"]),
            di[pf + "joint_vel"],
        ]

        # Add in top pos
        di[pf + "top_pos"] = np.array(self.sim.data.site_xpos[self.rod_top_id])
        robot_states.extend([di[pf + "top_pos"]])

        di[pf + "robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_limits(self):
        # Action limits based on controller limits
        low, high = self.controller.control_limits
        return low, high

    @property
    def torque_limits(self):
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 1]

        return low, high

    @property
    def action_dim(self):
        return self.controller.control_dim

    @property
    def dof(self):
        # Get the dof of the base robot model
        dof = super().dof
        return dof

    @property
    def js_energy(self):
        raise Exception("No need to call this function!")

    @property
    def ee_ft_integral(self):
        raise Exception("No need to call this function!")

    @property
    def ee_force(self):
        raise Exception("No need to call this function!")

    @property
    def ee_torque(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_pose(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_quat(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_total_velocity(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_pos(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_orn(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_vel(self):
        raise Exception("No need to call this function!")

    @property
    def _right_hand_ang_vel(self):
        raise Exception("No need to call this function!")
