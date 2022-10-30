import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Rod(RobotModel):

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        super().__init__(xml_path_completion("robots/rod/rod.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 1

    @property
    def gripper(self):
        return "NoGripper"

    @property
    def default_controller_config(self):
        return "default_rod"

    @property
    def init_qpos(self):
        return np.array([0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0)
        }

    @property
    def arm_type(self):
        return "IsNotArm"

    @property
    def _joints(self):
        return ["joint1"]

    @property
    def _eef_name(self):
        return "has_no_hand"

    @property
    def _robot_base(self):
        return "base"

    @property
    def _actuators(self):
        return {
            "pos": [],  # No position actuators for panda
            "vel": [],  # No velocity actuators for panda
            "torq": ["torq_j1"]
        }

    @property
    def _contact_geoms(self):
        return ["link1_collision"]

    @property
    def _root(self):
        return "link0"

    @property
    def _links(self):
        return ["link1"]
