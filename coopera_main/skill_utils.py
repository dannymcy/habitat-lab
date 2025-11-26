import habitat_sim
import magnum as mn
import warnings
import logging
import io
import imageio
import sys
import glob
import gc
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig, TopRGBSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)

from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig, ActionConfig
from habitat.core.env import Env

from habitat.utils.humanoid_utils import MotionConverterSMPLX
from habitat.tasks.rearrange.actions.articulated_agent_action import ArticulatedAgentAction
from habitat.core.registry import registry
from gym import spaces

from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
import gzip
import json
import pandas as pd
from openpyxl.utils import get_column_letter
import csv
import copy
import random
import torch
import pathlib
import time, datetime

import cv2
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
from typing import Dict
import stopit

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)

from sentence_transformers import SentenceTransformer

from human_utils import *
from robot_utils import extract_frames


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@registry.register_task_action
class PickObjIdAction(ArticulatedAgentAction):
    @property
    def action_space(self):
        MAX_OBJ_ID = 1000
        return spaces.Dict({
            f"{self._action_arg_prefix}pick_obj_id": spaces.Discrete(MAX_OBJ_ID)
        })

    def step(self, *args, **kwargs):
        obj_id = kwargs[f"{self._action_arg_prefix}pick_obj_id"]
        # print(self.cur_grasp_mgr, obj_id)
        self.cur_grasp_mgr.snap_to_obj(obj_id)


@registry.register_task_action
class PlaceObjIdAction(ArticulatedAgentAction):
    @property
    def action_space(self):
        MAX_OBJ_ID = 1000
        return spaces.Dict({
            f"{self._action_arg_prefix}place_obj_id": spaces.Discrete(MAX_OBJ_ID)
        })

    def step(self, *args, **kwargs):
        obj_id = kwargs[f"{self._action_arg_prefix}place_obj_id"]
        self.cur_grasp_mgr.desnap()
        # if self.cur_grasp_mgr.is_grasped:
        #     self.cur_grasp_mgr.desnap()
        #     print("Released the currently grasped object.")
        # else:
        #     print("No object is currently grasped.")


@registry.register_task_action
class MoveEEAction(ArticulatedAgentAction):
    @property
    def action_space(self):
        MAX_OBJ_ID = 1000
        return spaces.Dict({
            f"{self._action_arg_prefix}pick_obj_id": spaces.Discrete(MAX_OBJ_ID)
        })

    def step(self, *args, **kwargs):
        obj_id = kwargs[f"{self._action_arg_prefix}pick_obj_id"]

        # Get the object position and EE position
        obj_pos = self._sim.get_rigid_object_manager().get_object_by_id(obj_id).translation
        ee_transform = self.cur_articulated_agent.ee_transform()  # Get EE transformation (rotation and translation)
        ee_pos = ee_transform.translation

        # Compute the direction vector from EE to the object
        direction_to_obj = obj_pos - ee_pos
        distance_to_obj = np.linalg.norm(direction_to_obj)

        # Normalize the direction vector
        if distance_to_obj > 1e-5:  # Avoid division by zero
            direction_to_obj_normalized = direction_to_obj / distance_to_obj
        else:
            direction_to_obj_normalized = np.zeros_like(direction_to_obj)

        # Define how far the EE should move towards the object per step
        move_step_size = min(distance_to_obj, 0.015)  # Move by constant step or stop when close

        # Move the EE towards the object
        new_ee_pos = ee_pos + direction_to_obj_normalized * move_step_size

        # Use inverse kinematics to calculate new joint positions for the arm
        joint_positions = self.cur_articulated_agent.calculate_ee_inverse_kinematics(new_ee_pos)

        # Set the new joint positions
        self.cur_articulated_agent.arm_joint_pos = joint_positions


@registry.register_task_action
class ResetEEAction(ArticulatedAgentAction):
    @property
    def action_space(self):
        return spaces.Dict({
            f"{self._action_arg_prefix}ee_target_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the stored target EE position
        self.ee_target_position = None

    def step(self, *args, **kwargs):
        # Retrieve the desired EE target position from the action arguments
        ee_target_position = kwargs.get(f"{self._action_arg_prefix}ee_target_position", None)
        if ee_target_position is not None:
            # Update the stored target EE position
            self.ee_target_position = np.array(ee_target_position, dtype=np.float32)

        if self.ee_target_position is None:
            print("No target EE position specified.")
            return

        # Get the current EE position
        ee_transform = self.cur_articulated_agent.ee_transform()
        ee_pos = ee_transform.translation

        # Compute the direction vector from current EE position to the target position
        direction_to_target = self.ee_target_position - ee_pos
        distance_to_target = np.linalg.norm(direction_to_target)

        # If the EE is close enough to the target, stop moving
        if distance_to_target < 1e-3:
            print("Reached target EE position.")
            return

        # Normalize the direction vector to get the movement direction
        if distance_to_target > 1e-5:
            direction_to_target_normalized = direction_to_target / distance_to_target
        else:
            direction_to_target_normalized = np.zeros_like(direction_to_target)

        # Define the step size for the EE movement per step
        move_step_size = min(distance_to_target, 0.015)  # Adjust the step size as needed

        # Compute the new EE position by moving towards the target
        new_ee_pos = ee_pos + direction_to_target_normalized * move_step_size

        # Use inverse kinematics to calculate new joint positions for the arm
        joint_positions = self.cur_articulated_agent.calculate_ee_inverse_kinematics(new_ee_pos)

        # Set the new joint positions
        self.cur_articulated_agent.arm_joint_pos = joint_positions


def make_videos(observations, output_dir):
    vut.make_video(
        observations,
        "agent_0_head_rgb",
        "color",
        os.path.join(output_dir, f"robot_scene_camera_rgb_video.mp4"),
        open_vid=False,  # Ensure this is set to False to prevent video from popping up
    )
    vut.make_video(
        observations,
        "agent_0_third_rgb",
        "color",
        os.path.join(output_dir, f"robot_third_rgb_video.mp4"),
        open_vid=False,
    )
    vut.make_video(
        observations,
        "agent_1_head_rgb",
        "color",
        os.path.join(output_dir, f"human_scene_camera_rgb_video.mp4"),
        open_vid=False, 
    )
    vut.make_video(
        observations,
        "agent_1_third_rgb",
        "color",
        os.path.join(output_dir, f"human_third_rgb_video.mp4"),
        open_vid=False,
    )
    # TODO: top perspective takes the most memory, aften OOM if resolution = 1024 x 768
    # when the observation and action.step is too long, causes OOM error
    # comment out for now, as this is only needed for demo
    # vut.make_video(
    #     observations,
    #     "agent_1_top_rgb",
    #     "color",
    #     os.path.join(output_dir, f"top_scene_camera_rgb_video.mp4"),
    #     open_vid=False,  # Ensure this is set to False to prevent video from popping up
    # )


def validate_observations_for_video(observations):
    """Explicitly check for required sensor keys"""
    
    # Check if observations list exists and has content
    if not observations or len(observations) == 0:
        print("ERROR: Empty observations list")
        return False
    
    # Check if first observation is a valid dict
    if not isinstance(observations[0], dict):
        print(f"ERROR: Invalid observation type: {type(observations[0])}")
        return False
    
    # Explicitly check for each required key
    required_keys = [
        "agent_0_head_rgb",
        "agent_0_third_rgb", 
        "agent_1_head_rgb",
        "agent_1_third_rgb"
    ]
    
    for key in required_keys:
        if key not in observations[0]:
            print(f"ERROR: Missing required sensor key: '{key}'")
            print(f"Available keys: {list(observations[0].keys())}")
            return False
    
    return True


def make_sim_cfg(agent_dict, scene_id, collab_type=None):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    # Set up an example scene
    # sim_cfg.scene = os.path.join(data_path, f"hab3_bench_assets/hab3-hssd/scenes/{scene_id}.scene_instance.json")  # This line does not matter
    # sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    if collab_type == 1:
        sim_cfg.scene = os.path.join(data_path, f"scene_datasets/hssd-hab/scenes_dynamic/{scene_id}.scene_instance.json")
    elif collab_type == 2:
        sim_cfg.scene = os.path.join(data_path, f"scene_datasets/hssd-hab/scenes/{scene_id}.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]
    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg


def make_hab_cfg(agent_dict, action_dict, scene_id, collab_type=None):
    sim_cfg = make_sim_cfg(agent_dict, scene_id, collab_type=collab_type)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    # Need to update the newest version of hssd-hab so that all object instances are included
    # dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path=f"data/hab3_bench_assets/episode_datasets/{scene_id}.json.gz")  # This decides which scene to select and how to put the objects
    if collab_type == 1:
        dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path=f"data/scene_datasets/hssd-hab/episode_datasets_dynamic/{scene_id}.json.gz")
    elif collab_type == 2:
         dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path=f"data/scene_datasets/hssd-hab/episode_datasets/{scene_id}.json.gz")
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg


def init_rearrange_env(agent_dict, action_dict, scene_id, collab_type=None):
    hab_cfg = make_hab_cfg(agent_dict, action_dict, scene_id, collab_type=collab_type)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)


def create_agent_action(agent_dict, scene_id, collab_type=None):
    # Define the action configurations for the actions
    action_dict = {
        "oracle_magic_grasp_action": ArmActionConfig(type="MagicGraspAction"),
        "base_velocity_action": BaseVelocityActionConfig(),
        "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", spawn_max_dist_to_obj=1.0),
        "pick_obj_id_action": ActionConfig(type="PickObjIdAction"),
        "place_obj_id_action": ActionConfig(type="PlaceObjIdAction"),
        "move_EE_action": ActionConfig(type="MoveEEAction"),
        "reset_EE_action": ActionConfig(type="ResetEEAction"),
        "humanoid_joint_action": HumanoidJointActionConfig(),
        "humanoid_navigate_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                                                        motion_control="human_joints",
                                                        spawn_max_dist_to_obj=1.0),
        "humanoid_pick_obj_id_action": HumanoidPickActionConfig(type="HumanoidPickObjIdAction"),
        "humanoid_place_obj_id_action": HumanoidPickActionConfig(type="HumanoidPlaceObjIdAction")
    }

    # Create a multi-agent action dictionary
    multi_agent_action_dict = {}
    for action_name, action_config in action_dict.items():
        for agent_id in sorted(agent_dict.keys()):
            # if action_name == "humanoid_navigate_action" and agent_id == "agent_0": continue
            # if action_name == "oracle_coord_action" and agent_id == "agent_1": continue
            multi_agent_action_dict[f"{agent_id}_{action_name}"] = action_config

    # Initialize the environment with the agent dictionary and the multi-agent action dictionary
    env = init_rearrange_env(agent_dict, multi_agent_action_dict, scene_id, collab_type=collab_type)

    # The environment contains a pointer to a Habitat simulator
    # print(env._sim)

    # We can query the actions available, and their action space
    # for action_name, action_space in env.action_space.items():
    #     print(action_name, action_space)

    return env


def calculate_bounding_box_size(bounding_box):
    # bounding_box.min and bounding_box.max exist and are vectors
    min_point = bounding_box.min
    max_point = bounding_box.max
    
    # Calculate the size along each dimension
    width = max_point.x - min_point.x
    height = max_point.y - min_point.y
    depth = max_point.z - min_point.z
    
    return width, height, depth  # not sure about the order


def interpolate_points(start, end, steps):
    return [start + (end - start) * (i / steps) for i in range(1, steps + 1)]


def map_rooms_to_bounds(semantics_file):
    with open(semantics_file, 'r') as file:
        data = json.load(file)
    
    room_bounds_mapping = {}
    
    for region in data["region_annotations"]:
        room_name = region["name"]
        min_bounds = region["min_bounds"]
        max_bounds = region["max_bounds"]
        
        room_bounds_mapping[room_name] = {
            "min_bounds": min_bounds,
            "max_bounds": max_bounds
        }
    return room_bounds_mapping


def create_static_obj_trans_dict(static_obj_handle_list, static_obj_id_list, static_obj_trans_list, static_obj_bb_list, instance_file, object_mapping, static_categories):
    """
    Create a dictionary with actual object names and their translations, only if they belong to specified static categories.
    """
    object_translation_dict = {}
    # TODO: multiple same objects will be added as a single instance, but probably fine
    for i, template_name in enumerate(static_obj_handle_list):
        if template_name in object_mapping:
            actual_name = object_mapping[template_name]['name']
            if pd.isna(actual_name): actual_name = object_mapping[template_name]['wnsynsetkey']
            super_category = object_mapping[template_name]['super_category']
            # if super_category in static_categories:
            if True:
                object_translation_dict[actual_name] = [static_obj_id_list[i], static_obj_trans_list[i], static_obj_bb_list[i]]
    return object_translation_dict
    

def load_object_mapping(csv_path):
    """
    Load the CSV file and create a mapping from id to actual object name and super_category.
    """
    objects_df = pd.read_csv(csv_path)
    object_mapping = {}
    for _, row in objects_df.iterrows():
        object_mapping[row['id']] = {
            'name': row['name'],
            'wnsynsetkey': row['wnsynsetkey'],
            'super_category': row['super_category']
        }
    return object_mapping


def find_closest_room(object_translation, room_bounds_mapping):
    """
    Find the closest room to the object based on its translation.
    """
    closest_room = None
    min_distance = float('inf')
    closest_center = None

    for room, bounds in room_bounds_mapping.items():
        min_bounds = bounds["min_bounds"]
        max_bounds = bounds["max_bounds"]
        distance = 0
        center_x = (min_bounds[0] + max_bounds[0]) / 2
        center_y = (min_bounds[1] + max_bounds[1]) / 2
        center_z = (min_bounds[2] + max_bounds[2]) / 2

        # Calculate distance to the nearest boundary on each axis
        for i in range(3):
            if object_translation[i] < min_bounds[i]:
                distance += (min_bounds[i] - object_translation[i]) ** 2
            elif object_translation[i] > max_bounds[i]:
                distance += (object_translation[i] - max_bounds[i]) ** 2

        distance = distance ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_room = room
            closest_center = (center_x, 0, center_z)

    return closest_room, closest_center


def map_objects_to_rooms(object_trans_dict, room_bounds_mapping):
    """
    Map each object to the room it is located in.
    """
    def get_room_for_object(object_translation, room_bounds_mapping):
        """
        Determine which room the object is in based on its translation.
        """
        for room, bounds in room_bounds_mapping.items():
            min_bounds = bounds["min_bounds"]
            max_bounds = bounds["max_bounds"]
            
            # TODO: archs such as doors and windows cannot be mapped to rooms
            if (min_bounds[0] <= object_translation[0] <= max_bounds[0] and
                # min_bounds[1] <= object_translation[1] <= max_bounds[1] and  # height is ignored
                min_bounds[2] <= object_translation[2] <= max_bounds[2]):
                return room
        return None
    
    obj_room_mapping = {}
    for obj_name, translation in object_trans_dict.items():
        room = get_room_for_object(translation[1], room_bounds_mapping)
        if room:
            obj_room_mapping[obj_name] = [translation[0], room]
        else:
            closest_room, _ = find_closest_room(translation[1], room_bounds_mapping)
            if closest_room:
                obj_room_mapping[obj_name] = [translation[0], closest_room]
    
    return obj_room_mapping


def map_single_object_to_room(object_translation, room_bounds_mapping):
    """
    Determine which room the object is in based on its translation and return the center of that room.
    """
    for room, bounds in room_bounds_mapping.items():
        min_bounds = bounds["min_bounds"]
        max_bounds = bounds["max_bounds"]
        
        # Check if the object is within the room bounds (ignoring height)
        if (min_bounds[0] <= object_translation[0] <= max_bounds[0] and
            # min_bounds[1] <= object_translation[1] <= max_bounds[1] and  # height is ignored
            min_bounds[2] <= object_translation[2] <= max_bounds[2]):
            # Calculate the center of the room
            center_x = (min_bounds[0] + max_bounds[0]) / 2
            center_y = (min_bounds[1] + max_bounds[1]) / 2
            center_z = (min_bounds[2] + max_bounds[2]) / 2
            return room, (center_x, 0, center_z)
    
    return find_closest_room(object_translation, room_bounds_mapping)


def select_pick_place_obj(env, scene_id, pick_obj_idx, place_obj_idx, collab_type=None):
    # semantics_file = os.path.join(data_path, f"hab3_bench_assets/hab3-hssd/semantics/scenes/{scene_id}.semantic_config.json")
    # instance_file = os.path.join(data_path, f"hab3_bench_assets/hab3-hssd/scenes/{scene_id}.scene_instance.json")
    semantics_file = os.path.join(data_path, f"scene_datasets/hssd-hab/semantics/scenes/{scene_id}.semantic_config.json")
    if collab_type == 1:
        instance_file = os.path.join(data_path, f"scene_datasets/hssd-hab/scenes_dynamic/{scene_id}.scene_instance.json")
    elif collab_type == 2:
        instance_file = os.path.join(data_path, f"scene_datasets/hssd-hab/scenes/{scene_id}.scene_instance.json")
    object_csv_path = os.path.join(data_path,'scene_datasets/hssd-hab/semantics/objects.csv')

    static_categories = ["storage_furniture", "support_furniture", "seating_furniture", "floor_covering", 
                         "sleeping_furniture", "bathroom_fixtures", "mirror",
                         "large_kitchen_appliance", "large_appliance", "kitchen_bathroom_fixture", 
                         "vehicle", "heating_cooling", "medium_kitchen_appliance", "display", "arch", "curtain",
                         "small_kitchen_appliance"]

    room_dict = map_rooms_to_bounds(semantics_file)
    obj_mapping = load_object_mapping(object_csv_path)
    dynamic_obj_trans_dict = {}
    static_obj_handle_list, static_obj_id_list, static_obj_trans_list, static_obj_bb_list = [], [], [], []
    
    aom = env.sim.get_articulated_object_manager()
    rom = env.sim.get_rigid_object_manager()

    # We can query the articulated and rigid objects
    print("\nList of dynamic articulated objects:")
    for handle, ao in aom.get_objects_by_handle_substring().items():
        print(handle, "id", aom.get_object_id_by_handle(handle))

    print("\nList of dynamic rigid objects:")
    for handle, ro in rom.get_objects_by_handle_substring().items():
        rigid_obj = rom.get_object_by_id(
            ro.object_id
        )
        if ro.awake:
            print(handle, "id", ro.object_id)
            template_name = handle.split('_:')[0]
            trans = (rom.get_object_by_id(ro.object_id)).translation
            bb = (rom.get_object_by_id(ro.object_id)).aabb
            actual_name = obj_mapping[template_name]['name'] if template_name in obj_mapping else handle
            dynamic_obj_trans_dict[actual_name] = [ro.object_id, trans, bb]
        else:
            static_obj_handle_list.append(handle.split('_:')[0])
            static_obj_id_list.append(ro.object_id)
            static_obj_trans_list.append((rom.get_object_by_id(ro.object_id)).translation)
            static_obj_bb_list.append((rom.get_object_by_id(ro.object_id)).aabb)

    with open(instance_file, 'r') as f: instance_data = json.load(f)
    static_obj_trans_dict = create_static_obj_trans_dict(static_obj_handle_list, static_obj_id_list, static_obj_trans_list, static_obj_bb_list, instance_data, obj_mapping, static_categories)
    
    static_obj_room_mapping = map_objects_to_rooms(static_obj_trans_dict, room_dict)
    dynamic_obj_room_mapping = map_objects_to_rooms(dynamic_obj_trans_dict, room_dict)

    # Object locations are dynamic, which means they change if agents pick and place them
    pick_obj_id = env.sim.scene_obj_ids[pick_obj_idx]
    place_obj_id = env.sim.scene_obj_ids[place_obj_idx]
    pick_object = rom.get_object_by_id(pick_obj_id)
    place_object = rom.get_object_by_id(place_obj_id)
    pick_object_trans = pick_object.translation
    place_object_trans = place_object.translation

    return room_dict, static_obj_trans_dict, dynamic_obj_trans_dict, static_obj_room_mapping, dynamic_obj_room_mapping, aom, rom


def pick_up_human(env, observations, humanoid_controller, pick_obj_id, pick_object_trans):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)  # This line is important
    for _ in range(100):
        action_dict = {"action": ("agent_1_humanoid_pick_obj_id_action"), "action_args": {"agent_1_humanoid_pick_obj_id": pick_obj_id}}
        observations.append(env.step(action_dict)) 


def pick_up_robot(env, observations, pick_obj_id, pick_object_trans):
    ee_transform = env.sim.agents_mgr[0].articulated_agent.ee_transform()  # Get EE transformation (rotation and translation)
    ee_pos = ee_transform.translation

    # Move EE
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))
    for _ in range(100):
        action_dict = {"action": ("agent_0_move_EE_action"), "action_args": {"agent_0_pick_obj_id": pick_obj_id}}
        observations.append(env.step(action_dict)) 

    # Pick object
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))
    for _ in range(100):
        action_dict = {"action": ("agent_0_pick_obj_id_action"), "action_args": {"agent_0_pick_obj_id": pick_obj_id}}
        observations.append(env.step(action_dict)) 

    # Reset EE
    for _ in range(100):
        action_dict = {"action": ("agent_0_reset_EE_action"), "action_args": {"agent_0_ee_target_position": ee_pos}}
        observations.append(env.step(action_dict)) 


def place_robot(env, observations, place_obj_id, place_object_trans):
    ee_transform = env.sim.agents_mgr[0].articulated_agent.ee_transform()  # Get EE transformation (rotation and translation)
    ee_pos = ee_transform.translation
    
    # Move EE
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))
    for _ in range(100):
        action_dict = {"action": ("agent_0_move_EE_action"), "action_args": {"agent_0_pick_obj_id": place_obj_id}}
        observations.append(env.step(action_dict))

    # Place object
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))
    for _ in range(100):
        action_dict = {"action": ("agent_0_place_obj_id_action"), "action_args": {"agent_0_place_obj_id": place_obj_id}}
        observations.append(env.step(action_dict)) 

    # Reset EE
    for _ in range(100):
        action_dict = {"action": ("agent_0_reset_EE_action"), "action_args": {"agent_0_ee_target_position": ee_pos}}
        observations.append(env.step(action_dict)) 


def rotate_robot_to_look_at_object(env, observations, action_dict, cur_robot_pos, cur_robot_rot, object_trans, interpolation_factor=0.1, angle_threshold=1e-3):
    """
    Gradually rotate the robot to look at the object by adjusting its base rotation.

    :param env: The environment with the robot.
    :param observations: List to store observations.
    :param action_dict: Actions to take.
    :param cur_robot_pos: Current position of the robot.
    :param cur_robot_rot: Current yaw angle of the robot in radians.
    :param object_trans: Position of the object to look at.
    :param interpolation_factor: Controls rotation speed.
    :param angle_threshold: Threshold to stop rotation.
    :return: The updated robot yaw angle.
    """
    # TODO: This function has issue. Sometimes the robot will face opposite direction to the object.
    # Suspect the issue is the position of the robot is not accurate during path planning.
    # Deactivate this function for now.
    
    # Calculate direction vector from robot to object (projected onto XZ plane)
    direction_to_object = object_trans - cur_robot_pos
    direction_to_object[1] = 0  # Ignore Y-axis for yaw rotation

    norm = np.linalg.norm(direction_to_object)
    if norm == 0:
        return cur_robot_rot  # Robot is at the object's position

    direction_to_object_normalized = direction_to_object / norm

    # Compute the desired yaw angle (rotation around Y-axis)
    desired_yaw_angle = np.arctan2(-direction_to_object_normalized[0], direction_to_object_normalized[2])

    # Normalize desired_yaw_angle to [-π, π]
    desired_yaw_angle = (desired_yaw_angle + np.pi) % (2 * np.pi) - np.pi

    # Current yaw angle
    current_yaw_angle = cur_robot_rot  # Scalar value in radians

    # Compute the initial angle difference and normalize it
    angle_difference = (desired_yaw_angle - current_yaw_angle + np.pi) % (2 * np.pi) - np.pi

    # Begin rotating the robot towards the desired yaw angle
    while True:
        # Check if the angle difference is within the threshold
        if np.abs(angle_difference) < angle_threshold:
            break

        # Compute the interpolated yaw angle
        interpolated_yaw_angle = current_yaw_angle + interpolation_factor * angle_difference

        # Normalize interpolated_yaw_angle to [-π, π]
        interpolated_yaw_angle = (interpolated_yaw_angle + np.pi) % (2 * np.pi) - np.pi

        # Update the robot's base rotation (yaw angle)
        env.sim.agents_mgr[0].articulated_agent.base_rot = interpolated_yaw_angle
        observations.append(env.step(action_dict))

        # Update current_yaw_angle for the next iteration
        current_yaw_angle = interpolated_yaw_angle

        # Recompute the angle difference based on the new current_yaw_angle
        angle_difference = (desired_yaw_angle - current_yaw_angle + np.pi) % (2 * np.pi) - np.pi

    return current_yaw_angle


def walk_to_robot(env, observations, predicate_idx, humanoid_controller, object_trans, object_bb, obj_trans_dict, room_dict):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    # TODO: sometimes the articulated agents are in collision with the scene during path planning
    # causing [16:46:35:943864]:[Error]:[Nav] PathFinder.cpp(1324)::getRandomNavigablePointInCircle : Failed to getRandomNavigablePoint.  Try increasing max tries if the navmesh is fine but just hard to sample from here
    # the checking logic is modified in habitat/tasks/rearrange/actions/oracle_nav_actions: place_agent_at_dist_from_pos --> habitat/tasks/rearrange/utils: _get_robot_spawns;
    original_object_trans = object_trans
    original_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    initial_observations_length = len(observations)
    obj_room, room_trans = map_single_object_to_room(object_trans, room_dict)

    if predicate_idx == 0:
        env.sim.agents_mgr[0].articulated_agent.base_pos = mn.Vector3(room_trans)
        env.sim.agents_mgr[1].articulated_agent.base_pos = env.sim.pathfinder.get_random_navigable_point_near(circle_center=mn.Vector3(room_trans), radius=8.0, island_index=-1)
    original_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    original_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)  # This line is important

    # Walk towards the object to place
    robot_displ = np.inf
    prev_robot_displ = -np.inf
    robot_angdiff = np.inf
    prev_robot_angdiff = -np.inf
    prev_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    prev_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    width, height, depth = calculate_bounding_box_size(object_bb)
    robot_threshold = min(width, depth) / 2.5  # this is hard to adjust
    
    while robot_displ > robot_threshold or robot_angdiff > 1e-3:  # TODO: change from robot_threshold of 1e-9 to 1e-3 avoids the OOM issue 
        prev_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        prev_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot

        action_dict = {
            "action": ("agent_0_oracle_coord_action"),  
            "action_args": {
                "agent_0_oracle_nav_lookat_action": object_trans,
                "agent_0_mode": 1
            }
        }
        
        observations.append(env.step(action_dict))
    
        robot_room, room_trans = map_single_object_to_room(env.sim.agents_mgr[0].articulated_agent.base_pos, room_dict)
        if obj_room != robot_room and robot_displ <= robot_threshold:
            del observations[initial_observations_length:]
            env.sim.agents_mgr[0].articulated_agent.base_pos = original_robot_pos
            env.sim.agents_mgr[0].articulated_agent.base_rot = original_robot_rot         
            object_trans = env.sim.pathfinder.get_random_navigable_point_near(circle_center=original_object_trans, radius=robot_threshold, island_index=-1)
            # vec_sample_obj = original_object_trans - sample
        
        if prev_robot_displ == robot_displ and prev_robot_angdiff == robot_angdiff:
            robot_threshold += 0.1

        prev_robot_displ = robot_displ
        prev_robot_angdiff = robot_angdiff
        cur_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        cur_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
        robot_displ = (cur_robot_pos - object_trans).length()  # robot_displ = (cur_robot_pos - prev_robot_pos).length()
        robot_angdiff = np.inf if (obj_room != robot_room and robot_displ <= robot_threshold) else np.abs(cur_robot_rot - prev_robot_rot)

    # time.sleep(2)
    # cur_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    # cur_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    # rotate_robot_to_look_at_object(env, observations, action_dict, cur_robot_pos, cur_robot_rot, object_trans)

    return original_robot_pos


# This version uses strict constraints for path planning.
def walk_to(env, observations, predicate_idx, humanoid_controller, object_trans, object_bb, obj_trans_dict, room_dict):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    # TODO: sometimes the articulated agents are in collision with the scene during path planning
    # causing [16:46:35:943864]:[Error]:[Nav] PathFinder.cpp(1324)::getRandomNavigablePointInCircle : Failed to getRandomNavigablePoint.  Try increasing max tries if the navmesh is fine but just hard to sample from here
    # the checking logic is modified in habitat/tasks/rearrange/actions/oracle_nav_actions: place_agent_at_dist_from_pos --> habitat/tasks/rearrange/utils: _get_robot_spawns;
    original_object_trans = object_trans
    initial_observations_length = len(observations)
    obj_room, room_trans = map_single_object_to_room(object_trans, room_dict)

    if predicate_idx == 0:
        env.sim.agents_mgr[1].articulated_agent.base_pos = mn.Vector3(room_trans)
        env.sim.agents_mgr[0].articulated_agent.base_pos = env.sim.pathfinder.get_random_navigable_point_near(circle_center=mn.Vector3(room_trans), radius=8.0, island_index=-1)
    original_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    original_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    original_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
    original_human_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)  # This line is important

    # Walk towards the object to place
    human_displ = np.inf
    prev_human_displ = -np.inf
    human_angdiff = np.inf
    prev_human_angdiff = -np.inf
    prev_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
    prev_human_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
    width, height, depth = calculate_bounding_box_size(object_bb)
    human_threshold = min(width, depth) / 2.5  # this is hard to adjust
    robot_threshold_init, robot_threshold_sec = 3.0, 2.0  # this is hard to adjust

    success = True
    MAX_ITERATIONS = 1800  # Hard limit to prevent infinite loops (30 seconds with 60 FPS)
    counter = 0

    while human_displ > human_threshold or human_angdiff > 1e-3:  # TODO: change from human_threshold of 1e-9 to 1e-3 avoids the OOM issue 
        prev_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        prev_human_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
        prev_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        prev_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot

        if (env.sim.agents_mgr[0].articulated_agent.base_pos - prev_human_pos).length() > robot_threshold_init:
            action_dict = {
                "action": ("agent_1_humanoid_navigate_action", "agent_0_oracle_coord_action"),  
                "action_args": {
                    "agent_1_oracle_nav_lookat_action": object_trans,
                    "agent_1_mode": 1,
                    "agent_0_oracle_nav_lookat_action": object_trans,
                    "agent_0_mode": 1
                }
            }
        else:
            action_dict = {
                "action": ("agent_1_humanoid_navigate_action"),  
                "action_args": {
                    "agent_1_oracle_nav_lookat_action": object_trans,
                    "agent_1_mode": 1
                }
            }
        
        
        observations.append(env.step(action_dict))

        human_room, room_trans = map_single_object_to_room(env.sim.agents_mgr[1].articulated_agent.base_pos, room_dict)
        if obj_room != human_room and human_displ <= human_threshold:
            del observations[initial_observations_length:]
            env.sim.agents_mgr[0].articulated_agent.base_pos = original_robot_pos
            env.sim.agents_mgr[0].articulated_agent.base_rot = original_robot_rot         
            env.sim.agents_mgr[1].articulated_agent.base_pos = original_human_pos
            env.sim.agents_mgr[1].articulated_agent.base_rot = original_human_rot
            object_trans = env.sim.pathfinder.get_random_navigable_point_near(circle_center=original_object_trans, radius=human_threshold, island_index=-1)
            # vec_sample_obj = original_object_trans - sample
        
        if prev_human_displ == human_displ and prev_human_angdiff == human_angdiff:
            human_threshold += 0.1

        prev_human_displ = human_displ
        prev_human_angdiff = human_angdiff
        cur_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        cur_human_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
        human_displ = (cur_human_pos - object_trans).length()  # human_displ = (cur_human_pos - prev_human_pos).length()
        human_angdiff = np.inf if (obj_room != human_room and human_displ <= human_threshold) else np.abs(cur_human_rot - prev_human_rot)

        counter += 1
        if counter >= MAX_ITERATIONS:
            success = False
            del observations[initial_observations_length:]
            env.sim.agents_mgr[0].articulated_agent.base_pos = original_robot_pos
            env.sim.agents_mgr[0].articulated_agent.base_rot = original_robot_rot         
            env.sim.agents_mgr[1].articulated_agent.base_pos = original_human_pos
            env.sim.agents_mgr[1].articulated_agent.base_rot = original_human_rot
            return success, original_human_pos

    robot_room, room_trans = map_single_object_to_room(env.sim.agents_mgr[0].articulated_agent.base_pos, room_dict)
    robot_human_displ = max((env.sim.agents_mgr[0].articulated_agent.base_pos - cur_human_pos).length() - robot_threshold_sec, 0)
    while ((env.sim.agents_mgr[0].articulated_agent.base_pos - cur_human_pos).length() > robot_human_displ or np.abs(env.sim.agents_mgr[0].articulated_agent.base_rot - prev_robot_rot) > 1e-3) or robot_room != obj_room:
        prev_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        prev_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
        
        action_dict = {
            "action": ("agent_0_oracle_coord_action"),  
            "action_args": {
                "agent_0_oracle_nav_lookat_action": cur_human_pos,
                "agent_0_mode": 1
            }
        }
        observations.append(env.step(action_dict))

        if prev_robot_pos == env.sim.agents_mgr[0].articulated_agent.base_pos and robot_room == obj_room: break
        robot_room, room_trans = map_single_object_to_room(env.sim.agents_mgr[0].articulated_agent.base_pos, room_dict)
    
    if not validate_observations_for_video(observations):
        success = False
        del observations[initial_observations_length:]
        env.sim.agents_mgr[0].articulated_agent.base_pos = original_robot_pos
        env.sim.agents_mgr[0].articulated_agent.base_rot = original_robot_rot         
        env.sim.agents_mgr[1].articulated_agent.base_pos = original_human_pos
        env.sim.agents_mgr[1].articulated_agent.base_rot = original_human_rot
        print("Navigation failed: Invalid observations for video creation")
    
    return success, original_human_pos


@stopit.threading_timeoutable(default='timeout')
def walk_to_with_timeout(env, observations, predicate_idx, humanoid_controller, 
                         object_trans, object_bb, obj_trans_dict, room_dict):
    return walk_to(env, observations, predicate_idx, humanoid_controller,
                  object_trans, object_bb, obj_trans_dict, room_dict)


# https://aihabitat.org/docs/habitat-lab/habitat.datasets.rearrange.navmesh_utils.html
# This version uses very relaxed constraints for path planning.
# It is guaranteed to work (i.e., it never gets stuck while navigating), because the target location is relaxed to make planning easier.
# Intended for challenging scenes where planning is difficult.
def walk_to_relaxed(env, observations, predicate_idx, humanoid_controller, object_trans, object_bb, obj_trans_dict, room_dict):
    """
    Robust navigation function with multiple fallback strategies and error recovery
    """
    # Validate navmesh first
    if not env.sim.pathfinder.is_loaded:
        print("ERROR: No navmesh loaded, cannot navigate")
        return
    
    # Save original state
    original_object_trans = object_trans
    initial_observations_length = len(observations)
    obj_room, room_trans = map_single_object_to_room(object_trans, room_dict)
    
    # Initialize positions
    if predicate_idx == 0:
        # Try to place agents in room with validation
        room_pos = mn.Vector3(room_trans)
        if env.sim.pathfinder.is_navigable(room_pos):
            env.sim.agents_mgr[1].articulated_agent.base_pos = room_pos
        
        robot_spawn = env.sim.pathfinder.get_random_navigable_point_near(
            circle_center=room_pos, 
            radius=8.0, 
            island_index=-1
        )
        if robot_spawn is not None and not np.isnan(robot_spawn).any():
            env.sim.agents_mgr[0].articulated_agent.base_pos = robot_spawn
    
    # Store original positions for recovery
    original_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    original_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    original_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
    original_human_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
    
    # Validate and snap target position
    snapped_target = env.sim.pathfinder.snap_point(object_trans)
    if snapped_target is not None and not np.isnan(snapped_target).any():
        # Check if path exists before attempting navigation
        path = habitat_sim.ShortestPath()
        path.requested_start = original_human_pos
        path.requested_end = snapped_target
        if env.sim.pathfinder.find_path(path):
            object_trans = snapped_target
    
    # Reset controller
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)
    
    # Calculate thresholds
    width, height, depth = calculate_bounding_box_size(object_bb)
    human_threshold = min(width, depth) / 2.5
    robot_threshold_init, robot_threshold_sec = 3.0, 2.0
    
    # Navigation parameters with safety limits
    MAX_ITERATIONS = 300  # Hard limit to prevent infinite loops (5 seconds with 60 FPS)
    STUCK_THRESHOLD = 20  # Steps without movement before considering stuck (0.3 second with 60 FPS)
    stuck_counter = 0
    iteration_count = 0
    last_valid_human_pos = original_human_pos
    
    # Main navigation loop for human
    while iteration_count < MAX_ITERATIONS:
        iteration_count += 1
        
        # Get current positions
        prev_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        prev_human_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
        
        # Check if reached target
        human_displ = (prev_human_pos - object_trans).length()
        if human_displ <= human_threshold:
            # Check angle difference
            human_angdiff = np.abs(prev_human_rot - env.sim.agents_mgr[1].articulated_agent.base_rot)
            if human_angdiff <= 1e-3:
                print(f"Human reached target in {iteration_count} iterations")
                break
        
        # Determine if robot should also move
        robot_human_dist = (env.sim.agents_mgr[0].articulated_agent.base_pos - prev_human_pos).length()
        
        if robot_human_dist > robot_threshold_init:
            action_dict = {
                "action": ("agent_1_humanoid_navigate_action", "agent_0_oracle_coord_action"),
                "action_args": {
                    "agent_1_oracle_nav_lookat_action": object_trans,
                    "agent_1_mode": 1,
                    "agent_0_oracle_nav_lookat_action": object_trans,
                    "agent_0_mode": 1
                }
            }
        else:
            action_dict = {
                "action": ("agent_1_humanoid_navigate_action"),
                "action_args": {
                    "agent_1_oracle_nav_lookat_action": object_trans,
                    "agent_1_mode": 1
                }
            }
        
        # Execute action with error handling
        try:
            observations.append(env.step(action_dict))
        except Exception as e:
            print(f"Navigation step failed: {e}")
            break
        
        # Check if human moved (stuck detection)
        cur_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        movement = (cur_human_pos - prev_human_pos).length()
        
        if movement < 0.001:  # Essentially no movement
            stuck_counter += 1
            
            if stuck_counter > STUCK_THRESHOLD:
                print(f"Human appears stuck after {stuck_counter} steps, attempting recovery")
                
                # Try to find alternative target position
                search_radius = human_threshold * (1 + stuck_counter / 10)  # Gradually increase search radius
                alternative_target = env.sim.pathfinder.get_random_navigable_point_near(
                    circle_center=original_object_trans,
                    radius=search_radius,
                    island_index=-1
                )
                
                if alternative_target is not None and not np.isnan(alternative_target).any():
                    object_trans = alternative_target
                    stuck_counter = 0  # Reset counter
                    print(f"Found alternative target at radius {search_radius}")
                else:
                    # Can't find alternative, increase threshold
                    human_threshold *= 1.1
                    stuck_counter = 0
                    print(f"Increased threshold to {human_threshold}")
        else:
            stuck_counter = 0
            last_valid_human_pos = cur_human_pos
        
        # Room validation
        human_room, _ = map_single_object_to_room(cur_human_pos, room_dict)
        if obj_room != human_room and human_displ <= human_threshold:
            print(f"Human entered wrong room, adjusting target")
            # Find a point in the correct room
            object_trans = env.sim.pathfinder.get_random_navigable_point_near(
                circle_center=original_object_trans,
                radius=human_threshold * 2,
                island_index=-1
            )
            if object_trans is None:
                print("Could not find valid target in correct room")
                break
    
    # Robot follow-up navigation with timeout
    cur_human_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
    robot_iterations = 0
    MAX_ROBOT_ITERATIONS = 200  # Hard limit to prevent infinite loops (3.3 seconds with 60 FPS)
    
    while robot_iterations < MAX_ROBOT_ITERATIONS:
        robot_iterations += 1
        
        cur_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        robot_room, _ = map_single_object_to_room(cur_robot_pos, room_dict)
        
        # Check if robot is close enough and in correct room
        robot_human_dist = (cur_robot_pos - cur_human_pos).length()
        if robot_human_dist <= robot_threshold_sec and robot_room == obj_room:
            print(f"Robot reached target in {robot_iterations} iterations")
            break
        
        # Robot navigation
        action_dict = {
            "action": ("agent_0_oracle_coord_action"),
            "action_args": {
                "agent_0_oracle_nav_lookat_action": cur_human_pos,
                "agent_0_mode": 1
            }
        }
        
        try:
            observations.append(env.step(action_dict))
        except Exception as e:
            print(f"Robot navigation failed: {e}")
            break
        
        # Check if robot is stuck
        new_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        if (new_robot_pos - cur_robot_pos).length() < 0.001:
            stuck_counter += 1
            if stuck_counter > 10:
                print("Robot appears stuck, stopping robot navigation")
                break
        else:
            stuck_counter = 0
    
    # Log final status
    if iteration_count >= MAX_ITERATIONS:
        print(f"WARNING: Reached max iterations ({MAX_ITERATIONS}) in walk_to")
    
    return original_human_pos


def move_hand_and_place(env, observations, humanoid_controller, place_obj_id, place_object_trans, max_reach=0.809):
    # Get the initial hand pose
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)
    offset =  env.sim.agents_mgr[1].articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
    hand_pose = env.sim.agents_mgr[1].articulated_agent.ee_transform(0).translation + offset
    hand_pose_original = hand_pose

    # Calculate the initial distance between the hand and the object to place
    initial_distance = np.linalg.norm(place_object_trans - hand_pose_original)
    # print(initial_distance)

    if initial_distance <= max_reach:
        for hand_pose in interpolate_points(hand_pose_original, place_object_trans, 10):
            humanoid_controller.calculate_reach_pose(hand_pose, index_hand=0)

            # Get the new pose in the format expected by HumanoidJointAction
            new_pose = humanoid_controller.get_pose()
            action_dict = {
                "action": "agent_1_humanoid_joint_action",
                "action_args": {"agent_1_human_joints_trans": new_pose}
            }
            observations.append(env.step(action_dict))
    else:
        # Calculate the direction vector from the hand to the object
        direction_vector = place_object_trans - hand_pose_original
        direction_vector.y = 0  # Only consider x and z for horizontal movement
        horizontal_distance = np.linalg.norm(direction_vector)
        
        # Calculate new position within max_reach above the object
        if horizontal_distance == 0:
            new_hand_pose = mn.Vector3(place_object_trans)
            new_hand_pose.y += max_reach  # Move directly above
        else:
            # Scale direction vector to be within max reach distance
            scaled_vector = direction_vector * (max_reach / horizontal_distance)
            new_hand_pose = place_object_trans + scaled_vector
            new_hand_pose.y += max_reach - np.linalg.norm(scaled_vector)
        
        for hand_pose in interpolate_points(hand_pose_original, new_hand_pose, 10):
            humanoid_controller.calculate_reach_pose(hand_pose, index_hand=0)

            # Get the new pose in the format expected by HumanoidJointAction
            new_pose = humanoid_controller.get_pose()
            action_dict = {
                "action": "agent_1_humanoid_joint_action",
                "action_args": {"agent_1_human_joints_trans": new_pose}
            }
            observations.append(env.step(action_dict))

    # Place object 
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)
    for _ in range(100):
        action_dict = {"action": ("agent_1_humanoid_place_obj_id_action"), "action_args": {"agent_1_humanoid_place_obj_id": place_obj_id}}
        observations.append(env.step(action_dict)) 
    

def customized_humanoid_motion(env, observations, convert_helper, folder_dict, human_urdf_path, npy_file_folder_list, motion_pkl_path):
    # TODO: sometimes the articulated agents are violating the hold constraint,
    # causing AssertionError: Episode over, call reset before calling step
    # the checking logic is disabled in habitat/core/env: _update_step_stats --> habitat/tasks/rearrange/rearrange_task: _check_episode_is_active;
    create_motion_sets(npy_file_folder_list, human_urdf_path, sample_motion=os.path.splitext(os.path.basename(motion_pkl_path))[0], update=True)
    motion_npz_path = motion_pkl_path.replace('.pkl', '.npz')
    motion_folder = get_motion_pkl_path(motion_pkl_path, folder_dict)

    # human_rot is added to avoid the human agent looking at a different direction suddenly, when transitioning from rearrange to seq_pose controller
    # TODO: with the above implemented, sometimes the human will do motion facing an opposite direction, or do nothing
    # the above situation usually happens when human is very close to the scene objects/robot, so suspect collision is the reason
    convert_helper.convert_motion_file(
        motion_path=motion_npz_path,
        output_path=motion_npz_path.replace(".npz", ""),
        human_rot=env.sim.agents_mgr[1].articulated_agent.base_transformation,
        # reverse=(motion_folder==npy_file_folder_list[0])
        reverse=False
    )

    # Because we want the humanoid controller to generate a motion relative to the current agent, we need to set the reference pose
    humanoid_controller = HumanoidSeqPoseController(motion_pkl_path)
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)
    humanoid_controller.apply_base_transformation(env.sim.agents_mgr[1].articulated_agent.base_transformation)

    for _ in range(humanoid_controller.humanoid_motion.num_poses):
        # These computes the current pose and calculates the next pose
        humanoid_controller.calculate_pose()
        humanoid_controller.next_pose()
        
        # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "agent_1_humanoid_joint_action",
            "action_args": {"agent_1_human_joints_trans": new_pose}
        }
        observations.append(env.step(action_dict))


def execute_humanoid_1(env, observations, output_dir, humanoid_rearrange_controller, human_id, scene_id, day, time_, extracted_planning, motion_sets_list, obj_room_mapping, obj_trans_dict, room_dict, model, gpt=True):
    # TODO: Using sudo dmesg -T, the process is sometimes killed because OOM Killer. 
    # The reason is likely to be for some free-form motion, the robot planned path towards an object is never found / the robot is in collision with the scene, and increases computation overhead.
    # When rendering the videos, it causes OOM Killer.
    static_obj_room_mapping, dynamic_obj_room_mapping = obj_room_mapping[0], obj_room_mapping[1]
    static_obj_trans_dict, dynamic_obj_trans_dict = obj_trans_dict[0], obj_trans_dict[1]
    planning = extracted_planning[list(extracted_planning.keys())[0]]["Predicate_Acts"]

    for i, step in enumerate(planning):  # planning for each predicate
        if i != 0: continue  # the first predicate
        # initial_observations_length = len(observations)

        # search for static object to be placed
        place_object_id, place_object_trans, place_object_bb = None, None, None
        for name, (obj_id, trans, bb) in static_obj_trans_dict.items():
            if obj_id == step[0] and name == step[1]:
                place_object_id = obj_id
                place_object_trans = trans
                place_object_bb = bb
                break
            elif obj_id == step[0]:
                place_object_id = obj_id
                place_object_trans = trans
                place_object_bb = bb
                break
            elif name == step[1]:
                place_object_id = obj_id
                place_object_trans = trans
                place_object_bb = bb
                break

        # search for dynamic object to be picked
        pick_object_id, pick_object_trans, pick_object_bb = None, None, None
        for name, (obj_id, trans, bb) in dynamic_obj_trans_dict.items():
            if obj_id == step[2] and name == step[3]:
                pick_object_id = obj_id
                pick_object_trans = trans
                pick_object_bb = bb
                break
            elif obj_id == step[2]:
                pick_object_id = obj_id
                pick_object_trans = trans
                pick_object_bb = bb
                break
            elif name == step[3]:
                pick_object_id = obj_id
                pick_object_trans = trans
                pick_object_bb = bb
                break
        
        if place_object_id is None:  # the object is not found in the dict, because of mistake made by VLM
            _, (place_object_id, place_object_trans, place_object_bb) = most_similar_object(step[1], static_obj_trans_dict, model)
        
        if pick_object_id is None:  # the object is not found in the dict, because of mistake made by VLM
            _, (pick_object_id, pick_object_trans, pick_object_bb) = most_similar_object(step[3], dynamic_obj_trans_dict, model)

        result = walk_to_with_timeout(env, observations, i, humanoid_rearrange_controller,
                                    pick_object_trans, pick_object_bb, dynamic_obj_trans_dict, 
                                    room_dict, timeout=30)
        if result == 'timeout':
            print("walk_to timed out when picking")
            success_pick = False
        else:
            success_pick, _ = result
        if not success_pick: 
            observations[:] = []
            walk_to_relaxed(env, observations, i, humanoid_rearrange_controller, pick_object_trans, pick_object_bb, dynamic_obj_trans_dict, room_dict)

        pick_up_human(env, observations, humanoid_rearrange_controller, pick_object_id, pick_object_trans)

        if not success_pick:  # this means it is hard to do path planning in the scene
            walk_to_relaxed(env, observations, 999, humanoid_rearrange_controller, place_object_trans, place_object_bb, static_obj_trans_dict, room_dict)
        else:
            result = walk_to_with_timeout(env, observations, 999, humanoid_rearrange_controller,
                                        place_object_trans, place_object_bb, static_obj_trans_dict, 
                                        room_dict, timeout=30)
            if result == 'timeout':
                print("walk_to timed out when placing")
                success_place = False
            else:
                success_place, _ = result
            if not success_place: walk_to_relaxed(env, observations, 999, humanoid_rearrange_controller, place_object_trans, place_object_bb, static_obj_trans_dict, room_dict)
        
        move_hand_and_place(env, observations, humanoid_rearrange_controller, place_object_id, place_object_trans)
        print("step done")
        print()
        
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        if gpt:
            video_dir = os.path.join(output_dir, f"human/gpt_response/collaboration_1/{str(human_id).zfill(5)}/{scene_id}/{day}/{time_string}_{time_}")
        else:
            video_dir = os.path.join(output_dir, f"human/llama_response/collaboration_1/{str(human_id).zfill(5)}/{scene_id}/{day}/{time_string}_{time_}")
        os.makedirs(video_dir, exist_ok=True)

        make_videos(observations, video_dir)
        extract_frames(os.path.join(video_dir, f"robot_scene_camera_rgb_video.mp4"), os.path.join(video_dir, f"robot_scene_camera_rgb_video"))
        extract_frames(os.path.join(video_dir, f"human_third_rgb_video.mp4"), os.path.join(video_dir, f"human_third_rgb_video"))
        # del observations[initial_observations_length:]
        

def execute_humanoid_2(env, observations, convert_helper, folder_dict, motion_dict, human_urdf_path, output_dir, humanoid_rearrange_controller, human_id, scene_id, day, time_, extracted_planning, motion_sets_list, npy_file_folder_list, obj_room_mapping, obj_trans_dict, room_dict, model, gpt=True):
    # TODO: Using sudo dmesg -T, the process is sometimes killed because OOM Killer. 
    # The reason is likely to be for some free-form motion, the robot planned path towards an object is never found / the robot is in collision with the scene, and increases computation overhead.
    # When rendering the videos, it causes OOM Killer.
    static_obj_room_mapping, dynamic_obj_room_mapping = obj_room_mapping[0], obj_room_mapping[1]
    static_obj_trans_dict, dynamic_obj_trans_dict = obj_trans_dict[0], obj_trans_dict[1]
    planning = extracted_planning[list(extracted_planning.keys())[0]]["Predicate_Acts"]

    for i, step in enumerate(planning):  # planning for each predicate
        if i != 0: continue  # the first predicate
        # initial_observations_length = len(observations)

        obj_trans_dict_to_search = static_obj_trans_dict  # only dynamic object can be picked or placed
        object_trans, object_bb = None, None
        for name, (obj_id, trans, bb) in obj_trans_dict_to_search.items():
            if obj_id == step[1] and name == step[2]:
                object_trans = trans
                object_bb = bb
                break
            elif obj_id == step[1]:
                object_trans = trans
                object_bb = bb
                break
            elif name == step[2]:
                object_trans = trans
                object_bb = bb
                break
        
        if object_trans is None:  # the object is not found in the dict, because of mistake made by VLM
            _, (_, object_trans, object_bb) = most_similar_object(step[2], obj_trans_dict_to_search, model)

        result = walk_to_with_timeout(env, observations, i, humanoid_rearrange_controller,
                                    object_trans, object_bb, obj_trans_dict_to_search, 
                                    room_dict, timeout=30)
        if result == 'timeout':
            print("walk_to timed out")
            success = False
        else:
            success, _ = result
        if not success: walk_to_relaxed(env, observations, i, humanoid_rearrange_controller, object_trans, object_bb, obj_trans_dict_to_search, room_dict)

        if step[0] == 1:
            selected_motion = most_similar_motion(step[4], motion_sets_list, model)[0]
            customized_humanoid_motion(env, observations, convert_helper, folder_dict, human_urdf_path, npy_file_folder_list, get_motion_pkl_path(selected_motion, motion_dict))
            print()
            print(selected_motion)
        else:
            if step[0] == 2:
                pick_up_human(env, observations, humanoid_rearrange_controller, step[1], object_trans)
            elif step[0] == 3:
                move_hand_and_place(env, observations, humanoid_rearrange_controller, step[1], object_trans)
        print("step done")
        print()
        
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        if gpt:
            video_dir = os.path.join(output_dir, f"human/gpt_response/collaboration_2/{str(human_id).zfill(5)}/{scene_id}/{day}/{time_string}_{time_}")
        else:
            video_dir = os.path.join(output_dir, f"human/llama_response/collaboration_2/{str(human_id).zfill(5)}/{scene_id}/{day}/{time_string}_{time_}")
        os.makedirs(video_dir, exist_ok=True)
        with open(os.path.join(video_dir, f"{selected_motion}.txt"), 'w') as file: file.write(selected_motion)

        make_videos(observations, video_dir)
        extract_frames(os.path.join(video_dir, f"robot_scene_camera_rgb_video.mp4"), os.path.join(video_dir, f"robot_scene_camera_rgb_video"))
        extract_frames(os.path.join(video_dir, f"human_third_rgb_video.mp4"), os.path.join(video_dir, f"human_third_rgb_video"))
        # del observations[initial_observations_length:]
