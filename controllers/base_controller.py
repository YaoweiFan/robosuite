import abc
from collections.abc import Iterable
import numpy as np
import mujoco_py


class Controller(object, metaclass=abc.ABCMeta):
    """
    General controller interface.

    Requires reference to mujoco sim object, eef_name of specific robot, relevant joint_indexes to that robot, and
    whether an initial_joint is used for nullspace torques or not

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
    """
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
    ):

        # Actuator range
        self.actuator_min = actuator_range[0]
        self.actuator_max = actuator_range[1]

        # Attributes for scaling / clipping inputs to outputs
        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

        # Private property attributes
        self.control_dim = None
        self.output_min = None
        self.output_max = None
        self.input_min = None
        self.input_max = None

        # mujoco simulator state
        self.sim = sim
        self.model_timestep = self.sim.model.opt.timestep
        self.eef_name = eef_name
        self.joint_index = joint_indexes["joints"]
        self.qpos_index = joint_indexes["qpos"]
        self.qvel_index = joint_indexes["qvel"]

        # sensor state
        self.ft_name = self.eef_name[:9] + "ft_frame"
        self.fs_name = self.eef_name[:9] + "force_ee"
        self.ts_name = self.eef_name[:9] + "torque_ee"
        self.ft_pos = None
        self.ft_ori_mat = None
        self.eef_inertial_name = self.eef_name[:9] + "inertial_frame"
        self.eef_inertial_pos = None
        self.eef_inertial_ori_mat = None
        self.eef_ft = None
        self.eef_force = np.array([0, 0, 0])
        self.eef_torque = np.array([0, 0, 0])

        # robot states
        self.ee_pos = None
        self.ee_ori_mat = None
        self.ee_pos_vel = None
        self.ee_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None

        # dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None

        # Joint dimension
        self.joint_dim = len(joint_indexes["joints"])

        # Torques being outputted by the controller
        self.torques = None

        # Update flag to prevent redundant update calls
        self.new_update = True

        # Move forward one timestep to propagate updates before taking first update
        self.sim.forward()

        # Initialize controller by updating internal state and setting the initial joint, pos, and ori
        self.update()
        self.initial_joint = self.joint_pos
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

    @abc.abstractmethod
    def run_controller(self):
        """
        Abstract method that should be implemented in all subclass controllers, and should convert a given action
        into torques (pre gravity compensation) to be executed on the robot.
        Additionally, resets the self.new_update flag so that the next self.update call will occur
        """
        self.new_update = True

    def scale_action(self, action):
        """
        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """

        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def update(self, force=False):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
        self.new_update flag

        Args:
            force (bool): Whether to force an update to occur or not
        """

        # Only run update if self.new_update or force flag is set
        if self.new_update or force:
            self.sim.forward()

            self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])
            self.ee_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_name)].reshape([3, 3]))
            self.ee_pos_vel = np.array(self.sim.data.site_xvelp[self.sim.model.site_name2id(self.eef_name)])
            self.ee_ori_vel = np.array(self.sim.data.site_xvelr[self.sim.model.site_name2id(self.eef_name)])

            self.ft_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.ft_name)])
            self.ft_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.ft_name)].reshape([3, 3]))
            
            self.eef_inertial_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_inertial_name)])
            self.eef_inertial_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_inertial_name)].reshape([3, 3]))

            self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
            self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

            self.J_pos = np.array(self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
            self.J_ori = np.array(self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
            self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

            mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
            mujoco_py.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
            mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
            self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]

            # 末端受到的外力在笛卡尔空间的表示
            sensor_idx = np.sum(self.sim.model.sensor_dim[:self.sim.model.sensor_name2id(self.fs_name)])
            sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id(self.fs_name)]
            eef_force = np.array(self.sim.data.sensordata[sensor_idx: sensor_idx + sensor_dim])
            # eef_force = np.clip(eef_force, -20, 20)
            eef_force = 0.9 * self.eef_force + 0.1 * eef_force
            self.eef_force = eef_force
            sensor_idx = np.sum(self.sim.model.sensor_dim[:self.sim.model.sensor_name2id(self.ts_name)])
            sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id(self.ts_name)]
            eef_torque = np.array(self.sim.data.sensordata[sensor_idx: sensor_idx + sensor_dim])
            # eef_torque = np.clip(eef_torque, -20, 20)
            eef_torque = 0.9 * self.eef_torque + 0.1 * eef_torque
            self.eef_torque = eef_torque
            # gripper重力和外力 在gripper末端坐标系下的表示 --> 基坐标系
            rot1 = np.dot(np.linalg.inv(self.ft_ori_mat), self.ee_ori_mat)
            p1 = self.ee_pos - self.ft_pos
            p1 = np.dot(np.linalg.inv(self.ft_ori_mat), p1)
            cp1 = np.array([[0, -p1[2], p1[1]], [p1[2], 0, -p1[0]], [-p1[1], p1[0], 0]])
            ad1 = np.array(np.hstack([np.array(np.vstack([rot1, np.dot(cp1,rot1)])), np.array(np.vstack([np.zeros((3, 3)), rot1]))]))
            fsg = np.dot(ad1.transpose(), np.concatenate((-eef_force, -eef_torque)))
            fsg = np.array(np.concatenate((np.dot(self.ee_ori_mat, fsg[:3]), np.dot(self.ee_ori_mat, fsg[3:6]))))
            # gripper重力 在gripper末端坐标系下的表示 --> 基坐标系
            rot2 = np.dot(np.linalg.inv(self.eef_inertial_ori_mat), self.ee_ori_mat)
            p2 = self.ee_pos - self.eef_inertial_pos
            p2 = np.dot(np.linalg.inv(self.eef_inertial_ori_mat), p2)
            cp2 = np.array([[0, -p2[2], p2[1]], [p2[2], 0, -p2[0]], [-p2[1], p2[0], 0]])
            ad2 = np.array(np.hstack([np.array(np.vstack([rot2, np.dot(cp2,rot2)])), np.array(np.vstack([np.zeros((3, 3)), rot2]))]))
            fri = np.dot(np.linalg.inv(self.eef_inertial_ori_mat), np.array([0, 0, -0.5*10]))
            tri = np.dot(np.linalg.inv(self.eef_inertial_ori_mat), np.array([0, 0, 0]))
            fig = np.dot(ad2.transpose(), np.concatenate((fri, tri)))
            fig = np.array(np.concatenate((np.dot(self.ee_ori_mat, fig[:3]), np.dot(self.ee_ori_mat, fig[3:6]))))
            # 外力 在gripper末端坐标系下的表示 --> 基坐标系
            self.eef_ft = fsg - fig

            # Clear self.new_update
            self.new_update = False

    def update_base_pose(self, base_pos, base_ori):
        """
        Optional function to implement in subclass controllers that will take in @base_pos and @base_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers e.g. IK, which
        is based on pybullet and requires knowledge of simulator state deviations between pybullet and mujoco

        Args:
            base_pos (3-tuple): x,y,z position of robot base in mujoco world coordinates
            base_ori (4-tuple): x,y,z,w orientation or robot base in mujoco world coordinates
        """
        pass

    def update_initial_joints(self, initial_joints):
        """
        Updates the internal attribute self.initial_joints. This is useful for updating changes in controller-specific
        behavior, such as with OSC where self.initial_joints is used for determine nullspace actions

        This function can also be extended by subclassed controllers for additional controller-specific updates

        Args:
            initial_joints (Iterable): Array of joint position values to update the initial joints
        """
        self.initial_joint = np.array(initial_joints)
        self.update(force=True)
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

    def clip_torques(self, torques):
        """
        Clips the torques to be within the actuator limits

        Args:
            torques (Iterable): Torques to clip

        Returns:
            np.array: Clipped torques
        """
        return np.clip(torques, self.actuator_min, self.actuator_max)

    def reset_goal(self):
        """
        Resets the goal -- usually by setting to the goal to all zeros, but in some cases may be different (e.g.: OSC)
        """
        raise NotImplementedError

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    @property
    def torque_compensation(self):
        """
        Gravity compensation for this robot arm

        Returns:
            np.array: torques
        """
        return self.sim.data.qfrc_bias[self.qvel_index]

    @property
    def actuator_limits(self):
        """
        Torque limits for this controller

        Returns:
            2-tuple:

                - (np.array) minimum actuator torques
                - (np.array) maximum actuator torques
        """
        return self.actuator_min, self.actuator_max

    @property
    def control_limits(self):
        """
        Limits over this controller's action space, which defaults to input min/max

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        return self.input_min, self.input_max

    @property
    def name(self):
        """
        Name of this controller

        Returns:
            str: controller name
        """
        raise NotImplementedError


