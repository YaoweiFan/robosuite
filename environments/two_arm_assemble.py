from collections import OrderedDict
import numpy as np

from robosuite.environments.robot_env import RobotEnv

from robosuite.models.arenas import TableWithHoleArena
from robosuite.models.objects import TwoAxisPegObject
from robosuite.models.tasks import ManipulationTask, UniformRandomSampler
from robosuite.models.robots import check_bimanual

import robosuite.utils.transform_utils as transform


class TwoArmAssemble(RobotEnv):
    """
    This class corresponds to the assemble task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table (Default option)

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler instance): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        use_indicator_object (bool): if True, sets up an indicator object that
            is useful for debugging.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="single-arm-opposed",
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        # initialization_noise="default",
        initialization_noise=None,
        table_full_size=(0.8, 1.0, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        prepare_steps=3,
    ):
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.prepare_steps = prepare_steps

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                rotation=(-np.pi / 3, np.pi / 3),
            )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 3.0 is provided if the pot is lifted and is parallel within 30 deg to the table

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.5], per-arm component that is proportional to the distance between each arm and its
              respective pot handle, and exactly 0.5 when grasping the handle
              - Note that the agent only gets the lifting reward when flipping no more than 30 degrees.
            - Grasping: in {0, 0.25}, binary per-arm component awarded if the gripper is grasping its correct handle
            - Lifting: in [0, 1.5], proportional to the pot's height above the table, and capped at a certain threshold

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Returns:
            float: reward value
        """
        reward = 0

        # use a shaping reward
        if self.reward_shaping:
            _contacts_0_lf = len(list(
                self.find_contacts(
                    self.robots[0].gripper.important_geoms["left_finger"], ["assemble_peg"]
                )
            )) > 0
            _contacts_0_rf = len(list(
                self.find_contacts(
                    self.robots[0].gripper.important_geoms["right_finger"], ["assemble_peg"]
                )
            )) > 0
            _contacts_1_lf = len(list(
                self.find_contacts(
                    self.robots[1].gripper.important_geoms["left_finger"], ["assemble_peg"]
                )
            )) > 0
            _contacts_1_rf = len(list(
                self.find_contacts(
                    self.robots[1].gripper.important_geoms["right_finger"], ["assemble_peg"]
                )
            )) > 0

            # Reaching reward with Grasping
            if _contacts_0_lf and _contacts_0_rf:
                reward += 1.5 * (1 - np.tanh(10.0 * np.linalg.norm(self._left_peg_to_hole)))

            # Reaching reward with Grasping
            if _contacts_1_lf and _contacts_1_rf:
                reward += 1.5 * (1 - np.tanh(10.0 * np.linalg.norm(self._right_peg_to_hole)))

            if self.reward_scale is not None:
                reward *= self.reward_scale / 3.0

        return reward

    def _post_action(self):
        """
        Run any necessary visualization after running the action

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method

        """
        super()._post_action()

        reward = self.reward()
        success = False
        defeat = False
        timeout = False

        self.robot1_ft_out_of_range = \
            self.robots[0].ee_force[0] > 200 or self.robots[0].ee_force[0] < -200 or \
            self.robots[0].ee_force[1] > 200 or self.robots[0].ee_force[1] < -200 or \
            self.robots[0].ee_force[2] > 800 or self.robots[0].ee_force[2] < -800 or \
            self.robots[0].ee_torque[0] > 5 or self.robots[0].ee_torque[0] < -5 or \
            self.robots[0].ee_torque[1] > 5 or self.robots[0].ee_torque[1] < -5 or \
            self.robots[0].ee_torque[2] > 3 or self.robots[0].ee_torque[2] < -3

        self.robot2_ft_out_of_range = \
            self.robots[1].ee_force[0] > 200 or self.robots[1].ee_force[0] < -200 or \
            self.robots[1].ee_force[1] > 200 or self.robots[1].ee_force[1] < -200 or \
            self.robots[1].ee_force[2] > 800 or self.robots[1].ee_force[2] < -800 or \
            self.robots[1].ee_torque[0] > 5 or self.robots[1].ee_torque[0] < -5 or \
            self.robots[1].ee_torque[1] > 5 or self.robots[1].ee_torque[1] < -5 or \
            self.robots[1].ee_torque[2] > 3 or self.robots[1].ee_torque[2] < -3

        # done 的几个原因：时间超出限制、目标物体脱离抓手、力传感器示数超出限制
        self.done = (self.timestep >= self.horizon or
                     self._left_grab_error > 1e-3 or self._right_grab_error > 1e-3 or
                     self.robot1_ft_out_of_range or self.robot2_ft_out_of_range) and (not self.ignore_done)

        if self.done:
            # 明确 done 的原因
            if self._check_success():
                success = True
            elif self._left_grab_error > 1e-3 or self._right_grab_error > 1e-3 or \
                    self.robot1_ft_out_of_range or self.robot2_ft_out_of_range:
                defeat = True
            else:
                timeout = True

        return reward, self.done, {"success": success, "defeat": defeat, "timeout": timeout}

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi/2, -np.pi/2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = transform.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:   # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = TableWithHoleArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 1.0),
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.peg = TwoAxisPegObject(name="peg")
        self.mujoco_objects = OrderedDict([("peg", self.peg)])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena, 
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.mujoco_objects, 
            visual_objects=None, 
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.peg_body_id = self.sim.model.body_name2id("peg")
        self.left_peg_site_id = self.sim.model.site_name2id("leftp")
        self.right_peg_site_id = self.sim.model.site_name2id("rightp")
        
        self.left_hole_bottom_id = self.sim.model.site_name2id("hbleft")
        self.right_hole_bottom_id = self.sim.model.site_name2id("hbright")

        self.table_top_id = self.sim.model.site_name2id("table_orign")
        self.center_of_table_id = self.sim.model.site_name2id("table_orign")
        # pos_center_of_table = self.sim.data.site_xpos[self.center_of_table_id]

        # 得到左右抓取点位置
        self.left_grab_point_id = self.sim.model.site_name2id("left_grab_point")
        self.right_grab_point_id = self.sim.model.site_name2id("right_grab_point")
        # self.pot_center_id = self.sim.model.site_name2id("pot_center")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            # obj_pos, obj_quat = self.model.place_objects()
            self.model.place_objects()

        # reset() 的最后一步，收紧夹爪
        for i in range(self.prepare_steps):
            self._prepare_action(np.array([0, 0, 0, 1, 0, 0, 0, 1]))

        for robot in self.robots:
            robot.controller.has_initialized = True

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        # # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pr0 = self.robots[0].robot_model.naming_prefix
            pr1 = self.robots[1].robot_model.naming_prefix

            # position and rotation of object
            peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
            peg_quat = transform.convert_quat(
                self.sim.data.body_xquat[self.peg_body_id], to="xyzw"
            )
            di["peg_pos"] = peg_pos
            di["peg_quat"] = peg_quat

            di[pr0 + "peg_pos"] = np.array(self._left_peg_xpos)
            di[pr1 + "peg_pos"] = np.array(self._right_peg_xpos)
            di[pr0 + "peg_to_hole"] = np.array(self._left_peg_to_hole)
            di[pr1 + "peg_to_hole"] = np.array(self._right_peg_to_hole)

            di["object-state"] = np.concatenate(
                [
                    di["peg_pos"],
                    di["peg_quat"],
                    di[pr0 + "peg_pos"],
                    di[pr1 + "peg_pos"],
                    di[pr0 + "peg_to_hole"],
                    di[pr1 + "peg_to_hole"],
                ]
            )

        return di

    def _check_success(self):
        """
        Check if peg is successfully assembled

        Returns:
            bool: True if peg is assembled
        """
        return (np.sum(self._left_peg_to_hole**2) + np.sum(self._right_peg_to_hole**2) < 1e-4) and \
            self._left_grab_error <= 1e-3 and \
            self._right_grab_error <= 1e-3

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        robots = robots if type(robots) == list or type(robots) == tuple else [robots]
        if self.env_configuration == "single-arm-opposed" or self.env_configuration == "single-arm-parallel":
            # Specifically two robots should be inputted!
            is_bimanual = False
            if type(robots) is not list or len(robots) != 2:
                raise ValueError("Error: Exactly two single-armed robots should be inputted "
                                 "for this task configuration!")
        elif self.env_configuration == "bimanual":
            is_bimanual = True
            # Specifically one robot should be inputted!
            if type(robots) is list and len(robots) != 1:
                raise ValueError("Error: Exactly one bimanual robot should be inputted "
                                 "for this task configuration!")
        else:
            # This is an unknown env configuration, print error
            raise ValueError("Error: Unknown environment configuration received. Only 'bimanual',"
                             "'single-arm-parallel', and 'single-arm-opposed' are supported. Got: {}"
                             .format(self.env_configuration))

        # Lastly, check to make sure all inputted robot names are of their correct type (bimanual / not bimanual)
        for robot in robots:
            if check_bimanual(robot) != is_bimanual:
                raise ValueError("Error: For {} configuration, expected bimanual check to return {}; "
                                 "instead, got {}.".format(self.env_configuration, is_bimanual, check_bimanual(robot)))

    @property
    def _left_peg_xpos(self):
        """
        Grab the position of the left peg.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.left_peg_site_id]

    @property
    def _right_peg_xpos(self):
        """
        Grab the position of the right peg.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.right_peg_site_id]

    @property
    def _world_quat(self):
        """
        Grab the world orientation

        Returns:
            np.array: (x,y,z,w) world quaternion
        """
        return transform.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _eef0_xpos(self):
        """
        Grab the position of Robot 0's end effector.

        Returns:
            np.array: (x,y,z) position of EEF0
        """
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["left"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef1_xpos(self):
        """
        Grab the position of Robot 1's end effector.

        Returns:
            np.array: (x,y,z) position of EEF1
        """
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id])

    @property
    def _left_hole_xpos(self):
        """
        Grab the position of left hole.

        Returns:
            np.array: (x,y,z) position of left hole
        """
        return np.array(self.sim.data.site_xpos[self.left_hole_bottom_id])

    @property
    def _right_hole_xpos(self):
        """
        Grab the position of right hole.

        Returns:
            np.array: (x,y,z) position of right hole
        """
        return np.array(self.sim.data.site_xpos[self.right_hole_bottom_id])

    @property
    def _left_peg_to_hole(self):
        """
        Calculate vector from the left peg to the left hole.

        Returns:
            np.array: (dx,dy,dz) distance vector between left peg and left hole.
        """
        return self._left_peg_xpos - self._left_hole_xpos

    @property
    def _right_peg_to_hole(self):
        """
        Calculate vector from the right peg to the right hole.

        Returns:
            np.array: (dx,dy,dz) distance vector between right peg and right hole.
        """
        return self._right_peg_xpos - self._right_hole_xpos

    @property
    def _left_grab_error(self):
        """
        Calculate vector from the left peg to the left eef.

        Returns:
            np.array: (dx,dy,dz) distance vector between left peg and left eef.
        """
        return np.sum((self.sim.data.site_xpos[self.left_grab_point_id] - self._eef0_xpos) ** 2)

    @property
    def _right_grab_error(self):
        """
        Calculate vector from the right peg to the right eef.

        Returns:
            np.array: (dx,dy,dz) distance vector between right peg and right eef.
        """
        return np.sum((self.sim.data.site_xpos[self.right_grab_point_id] - self._eef1_xpos) ** 2)
