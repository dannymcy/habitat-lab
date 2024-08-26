import habitat_sim
import magnum as mn
import warnings
import logging
import io
import sys
import glob
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

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)

from sentence_transformers import SentenceTransformer

from habitat.gpt.prompts.utils import *
from human_utils import *
from robot_utils import *
from judge_utils import *


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
        print(self.cur_grasp_mgr, obj_id)
        self.cur_grasp_mgr.snap_to_obj(obj_id)


def make_sim_cfg(agent_dict, scene_id):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    # Set up an example scene
    sim_cfg.scene = os.path.join(data_path, f"hab3_bench_assets/hab3-hssd/scenes/{scene_id}.scene_instance.json")  # THis line does not matter
    sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    # sim_cfg.scene = os.path.join(data_path, "datasets/scene_datasets/hssd-hab/scenes/103997919_171031233.scene_instance.json")
    # sim_cfg.scene_dataset = os.path.join(data_path, "datasets/scene_datasets/hssd-hab/hab3-hssd.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]
    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg


def make_hab_cfg(agent_dict, action_dict, scene_id):
    sim_cfg = make_sim_cfg(agent_dict, scene_id)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    # Need to update the newest version of hssd-hab so that all object instances are included
    # dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path=f"data/hab3_bench_assets/episode_datasets/{scene_id}.json.gz")  # This decides which scene to select and how to put the objects
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")  # This decides which scene to select and how to put the objects
    # dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/datasets/hssd/rearrange/train/social_rearrange.json.gz")
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg


def init_rearrange_env(agent_dict, action_dict, scene_id):
    hab_cfg = make_hab_cfg(agent_dict, action_dict, scene_id)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)


def create_agent_action(agent_dict, scene_id):
    # Define the action configurations for the actions
    action_dict = {
        "oracle_magic_grasp_action": ArmActionConfig(type="MagicGraspAction"),
        "base_velocity_action": BaseVelocityActionConfig(),
        "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", spawn_max_dist_to_obj=1.0),
        "pick_obj_id_action": ActionConfig(type="PickObjIdAction"),
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
    env = init_rearrange_env(agent_dict, multi_agent_action_dict, scene_id)

    # The environment contains a pointer to a Habitat simulator
    # print(env._sim)

    # We can query the actions available, and their action space
    # for action_name, action_space in env.action_space.items():
    #     print(action_name, action_space)

    return env


def make_videos(output_dir):
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
    vut.make_video(
        observations,
        "agent_1_top_rgb",
        "color",
        os.path.join(output_dir, f"top_scene_camera_rgb_video.mp4"),
        open_vid=False,  # Ensure this is set to False to prevent video from popping up
    )


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
    # TODO: multiple same objects will be added as a single instance
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


def select_pick_place_obj(env, scene_id, pick_obj_idx, place_obj_idx):
    semantics_file = os.path.join(data_path, f"hab3_bench_assets/hab3-hssd/semantics/scenes/{scene_id}.semantic_config.json")
    instance_file = os.path.join(data_path, f"hab3_bench_assets/hab3-hssd/scenes/{scene_id}.scene_instance.json")
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


def pick_up_human(env, humanoid_controller, pick_obj_id, pick_object_trans):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)  # This line is important
    for _ in range(100):
        action_dict = {"action": ("agent_1_humanoid_pick_obj_id_action"), "action_args": {"agent_1_humanoid_pick_obj_id": pick_obj_id}}
        observations.append(env.step(action_dict)) 


def pick_up_robot(env, pick_obj_id, pick_object_trans):
    # Wait
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))
    for _ in range(100):
        action_dict = {"action": ("agent_0_pick_obj_id_action"), "action_args": {"agent_0_pick_obj_id": pick_obj_id}}
        # action_dict = {"action": ("agent_0_oracle_magic_grasp_action"), "action_args": {"grip_action": 1}}
        observations.append(env.step(action_dict)) 
        print(len(observations))


def walk_to_robot(env, object_trans):
    original_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    robot_displ = np.inf
    robot_angdiff = np.inf

    while robot_displ > 1e-6 or robot_angdiff > 1e-6:
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
        print(len(observations))

        cur_robot_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
        cur_robot_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
        robot_displ = (cur_robot_pos - prev_robot_pos).length()
        robot_angdiff = np.abs(cur_robot_rot - prev_robot_rot)

    return original_robot_pos


def walk_to(env, predicate_idx, humanoid_controller, object_trans, object_bb, obj_trans_dict, room_dict):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    # TODO: sometimes the articulated agents are in collision with the scene during path planning
    # causing [16:46:35:943864]:[Error]:[Nav] PathFinder.cpp(1324)::getRandomNavigablePointInCircle : Failed to getRandomNavigablePoint.  Try increasing max tries if the navmesh is fine but just hard to sample from here
    # as a dirty solution, the checking logic is modified in habitat/tasks/rearrange/actions/oracle_nav_actions: place_agent_at_dist_from_pos --> habitat/tasks/rearrange/utils: _get_robot_spawns;
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
    robot_threshold_init, robot_threshold_sec = 3., 2.0  # this is hard to adjust
    
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


def move_hand_and_place(env, humanoid_controller, place_obj_id, place_object_trans, max_reach=0.809):
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
    

def customized_humanoid_motion(env, convert_helper, folder_dict, motion_pkl_path):
    # TODO: sometimes the articulated agents are violating the hold constraint,
    # causing AssertionError: Episode over, call reset before calling step
    # as a dirty solution, the checking logic is disabled in habitat/core/env: _update_step_stats --> habitat/tasks/rearrange/rearrange_task: _check_episode_is_active;
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


def execute_humanoid_1(env, human_id, scene_id, time_, extracted_planning, motion_sets_list, obj_room_mapping, obj_trans_dict, room_dict):
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
            _, (_, object_trans, object_bb) = most_similar_object(step[2], obj_trans_dict_to_search, DEVICE)
        
        walk_to(env, i, humanoid_rearrange_controller, object_trans, object_bb, obj_trans_dict_to_search, room_dict)
        
        if step[0] == 1:
            selected_motion = most_similar_motion(step[4], motion_sets_list, DEVICE)[0]
            customized_humanoid_motion(env, convert_helper, folder_dict, get_motion_pkl_path(selected_motion, motion_dict))
            print()
            print(selected_motion)
        else:
            if step[0] == 2:
                pick_up_human(env, humanoid_rearrange_controller, step[1], object_trans)
            elif step[0] == 3:
                move_hand_and_place(env, humanoid_rearrange_controller, step[1], object_trans)
        print("step done")
        print()
        
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        video_dir = os.path.join(output_dir, f"human/{str(human_id).zfill(5)}/{scene_id}/{time_string}_{time_}")
        os.makedirs(video_dir, exist_ok=True)
        with open(os.path.join(video_dir, f"{selected_motion}.txt"), 'w') as file: file.write(selected_motion)

        # original_robot_pos = walk_to_robot(env, mn.Vector3([-6.37489, 0.55358, -5.71439]))
        # pick_up_robot(env, 176, mn.Vector3([-6.37489, 0.55358, -5.71439]))
        # walk_to_robot(env, env.sim.agents_mgr[1].articulated_agent.base_pos)

        make_videos(video_dir)
        extract_frames(os.path.join(video_dir, f"robot_scene_camera_rgb_video.mp4"), os.path.join(video_dir, f"robot_scene_camera_rgb_video"))
        extract_frames(os.path.join(video_dir, f"human_third_rgb_video.mp4"), os.path.join(video_dir, f"human_third_rgb_video"))
        # del observations[initial_observations_length:]



os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda:2"
set_seed(42)

# conda activate /hdd2/chenyang/Dynamic_Human_Robot_Value_Alignments/env
# CUDA_VISIBLE_DEVICES="1,2,3" python examples_main/main_qlora.py
if __name__ == "__main__":
    output_dir = os.path.join(data_path, "interactive_play_replays")
    os.makedirs(output_dir, exist_ok=True)

    # Define the robot agent configuration
    robot_agent_config = AgentConfig()
    robot_urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
    robot_agent_config.articulated_agent_urdf = robot_urdf_path
    robot_agent_config.articulated_agent_type = "FetchRobot"

    # Define the human agent configuration
    human_agent_config = AgentConfig()
    human_urdf_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0.urdf")
    human_agent_config.articulated_agent_urdf = human_urdf_path
    human_agent_config.articulated_agent_type = "KinematicHumanoid"
    human_agent_config.motion_data_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl")
    robot_agent_config.motion_data_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl")  # placeholder
    humanoid_rearrange_controller = HumanoidRearrangeController(human_agent_config.motion_data_path)

    # Convert custom motion npy to npz
    # npy_file_folder_list = [os.path.join(data_path, "humanoids/humanoid_data/haa500_motion"),  # Some motion makes the human upside down.
    #                         os.path.join(data_path, "humanoids/humanoid_data/humman_motion"),
    #                         os.path.join(data_path, "humanoids/humanoid_data/idea400_motion"),
    #                         os.path.join(data_path, "humanoids/humanoid_data/perform_motion")]
    npy_file_folder_list = [os.path.join(data_path, "humanoids/humanoid_data/all_motion")]
                            
    motion_sets_list, motion_dict, folder_dict, convert_helper = create_motion_sets(npy_file_folder_list, human_urdf_path, update=False)
    print()
    print()
    print(f"Motion List Size: {len(motion_sets_list)}")
    print(motion_sets_list)
    print()

    # Define sensors that will be attached to this agent
    # how to set camera: https://github.com/facebookresearch/habitat-lab/issues/1737
    robot_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
    }
    human_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
        "top_rgb": TopRGBSensorConfig()
    }

    scene_id = "103997919_171031233"
    agent_dict = {"agent_0": robot_agent_config, "agent_1": human_agent_config}
    env = create_agent_action(agent_dict, scene_id)
    env.reset()
    
    room_dict, static_obj_trans_dict, dynamic_obj_trans_dict, static_obj_room_mapping, dynamic_obj_room_mapping, aom, rom = select_pick_place_obj(env, scene_id, 0, 0)
    room_list = list(room_dict.keys())

    print()
    print(room_list)
    print()
    print(room_dict)
    print()
    print(f"Static Object Translation Dict Size: {len(static_obj_trans_dict)}")
    print(static_obj_trans_dict)
    print()
    print(f"Dynamic Object Translation Dict Size: {len(dynamic_obj_trans_dict)}")
    print(dynamic_obj_trans_dict)
    print()
    print(f"Static Object Room Mapping Size: {len(static_obj_room_mapping)}")
    print(static_obj_room_mapping)
    print()
    print(f"Dynamic Object Room Mapping Size: {len(dynamic_obj_room_mapping)}")
    print(dynamic_obj_room_mapping)
    print()

    # Communicating to ChatGPT-4 API
    temperature_dict = {
        "traits_summary": 0.25,
        "intention_proposal": 0.7,
        "predicates_proposal": 0.7,
        "predicates_reflection": 0.25,
        "intention_discovery": 0.25,
        "predicates_discovery": 0.25,
        "traits_inference": 0.25,
        "collaboration_approval": 0.25
    }
    # GPT-4 1106-preview is GPT-4 Turbo (https://openai.com/pricing)
    model_dict = {
        "traits_summary": "gpt-4o",
        "intention_proposal": "gpt-4o",
        "predicates_proposal": "gpt-4o",
        "predicates_reflection": "gpt-4o",
        "intention_discovery": "gpt-4o",
        "predicates_discovery": "gpt-4o",
        "traits_inference": "gpt-4o",
        "collaboration_approval": "gpt-4o"
    }


    profile_string_list, big_five_list = read_human_data_mypersonality(data_path)
    times = ['9 am', '10 am', '11 am', '12 pm', 
             '1 pm', '2 pm', '3 pm', '4 pm', 
             '5 pm', '6 pm', '7 pm', '8 pm', '9 pm'
    ]
    predicates_num = 5
    qlora_traits_intention_dir, qlora_temporal_intention_dir, qlora_traits_predicates_dir, qlora_temporal_predicates_dir = "", "", "", ""
    confidence_intention_threshold, confidence_predicate_threshold = 0.82, 0.65

    for i, (profile_string, big_five) in enumerate(zip(profile_string_list, big_five_list)):
        if i > 4: continue
        human_intentions_hist, human_predicates_hist = [], []
        profile_string = traits_summary_gpt4(data_path, i, scene_id, [profile_string, big_five], temperature_dict, model_dict, start_over=False)[0][1]

        robot_intentions_hist, robot_predicates_hist = [], []
        inferred_traits = ""
        inferred_traits_hist = []
        intention_approval_hist, tasks_approval_hist = [], []
        confidence_intention_hist, confidence_predicates_hist = [], []

        prev_qlora_traits_intention_dir, prev_qlora_temporal_intention_dir = qlora_traits_intention_dir, qlora_temporal_intention_dir
        prev_qlora_traits_predicates_dir, prev_qlora_temporal_predicates_dir = qlora_traits_predicates_dir, qlora_temporal_predicates_dir
        data_train_traits_intention, data_train_temporal_intention, data_train_traits_predicates, data_train_temporal_predicates = [], [], [], []

        vis_dir = pathlib.Path(data_path) / "gpt4_response" / "vis" / str(i).zfill(5) / scene_id
        vis_csv_path = vis_dir / "vis.xlsx"
        vis_txt_path = vis_dir / "vis.txt"
        os.makedirs(vis_dir, exist_ok=True)
        csv_data = []

        for j, time_ in enumerate(times):
            # if j != 0: continue
            # Human Proposing Intention
            # human_retrieved_intentions = retrieve_memory(f"Current time: {time_}", human_intentions_hist, times, time_, predicates_num, DEVICE, decay_factor=0.95, top_k=13, retrieve_type="intention")
            human_retrieved_intentions = human_intentions_hist
            human_retrieved_predicates = retrieve_memory(f"Current time: {time_}", human_predicates_hist, times, time_, predicates_num, DEVICE, decay_factor=0.95, top_k=5, retrieve_type="predicate")

            human_conversation_hist = intention_proposal_gpt4(data_path, i, scene_id, [j, time_], [human_retrieved_intentions, human_retrieved_predicates], room_list, [profile_string, big_five], temperature_dict, model_dict, start_over=False)
            _, gt_intention_sentence_list, human_sampled_static_obj_dict_list = sample_obj_by_similarity(human_conversation_hist, static_obj_room_mapping, DEVICE, top_k=30)
            _, sampled_motion_list = sample_motion_by_similarity(human_conversation_hist, motion_sets_list, DEVICE, top_k=5)
            gt_intention_sentence, human_sampled_static_obj_dict, sampled_motion_list = gt_intention_sentence_list[0], human_sampled_static_obj_dict_list[0], sampled_motion_list[0]
            human_intentions_hist.append(f"{time_}: {gt_intention_sentence}")

            # Human Proposing Predicates
            human_retrieved_predicates = retrieve_memory(f"Current time: {time_}. Intention: {gt_intention_sentence}", human_predicates_hist, times, time_, predicates_num, DEVICE, decay_factor=0.95, top_k=5, retrieve_type="predicate")

            human_conversation_hist = predicates_proposal_gpt4(data_path, i, scene_id, [j, time_], [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, start_over=False)
            human_conversation_hist = predicates_reflection_1_gpt4(data_path, i, scene_id, [j, time_], [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, start_over=False)
            human_conversation_hist = predicates_reflection_2_gpt4(data_path, i, scene_id, [j, time_], [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, start_over=False)

            human_thoughts, human_acts = extract_thoughts_and_acts(human_conversation_hist[-1][1], search_txt=" Reason_human:")
            if not human_thoughts: human_thoughts, human_acts = extract_thoughts_and_acts(human_conversation_hist[-1][1], search_txt="")
            human_predicates_hist.extend([f"{time_}.{k}: {human_thought}" for k, human_thought in enumerate(human_thoughts)])


            # =====================================================================================================================
            # query the observation of each of the agents
            # without this will cause: AssertionError: Episode over, call reset before calling step
            observations = env.reset()
            _, ax = plt.subplots(1, len(observations.keys()))
            for ind, name in enumerate(observations.keys()):
                ax[ind].imshow(observations[name])
                ax[ind].set_axis_off()
                ax[ind].set_title(name)

            env.reset()
            observations = []

            extracted_planning = extract_code("predicates_reflection_2", pathlib.Path(data_path) / "gpt4_response" / "human/predicates_reflection_2" / str(i).zfill(5) / scene_id, j)

            # if j == 0:
            #     execute_humanoid_1(env, i, scene_id, time_, extracted_planning, motion_sets_list, [static_obj_room_mapping, dynamic_obj_room_mapping], [static_obj_trans_dict, dynamic_obj_trans_dict], room_dict)


            # =====================================================================================================================
            # Robot Inferring Intentions
            video_dir_search_pattern = os.path.join(output_dir, f"human/{str(i).zfill(5)}/{scene_id}/*_{time_}")
            video_dir = glob.glob(video_dir_search_pattern)[0]
    
            robot_retrieved_intentions = robot_intentions_hist
            robot_retrieved_predicates = retrieve_memory(f"Current time: {time_}", robot_predicates_hist, times, time_, predicates_num, DEVICE, decay_factor=0.95, top_k=5, retrieve_type="predicate")

            robot_conversation_hist = intention_discovery_gpt4(data_path, i, scene_id, [j, time_], [os.path.join(video_dir, "robot_scene_camera_rgb_video"), os.path.join(video_dir, "human_third_rgb_video")], [robot_retrieved_intentions, robot_retrieved_predicates], inferred_traits, temperature_dict, model_dict, start_over=True)
            _, ps_intention_sentence_list, robot_sampled_static_obj_dict_list = sample_obj_by_similarity(robot_conversation_hist, static_obj_room_mapping, DEVICE, top_k=30)
            ps_intention_sentence, robot_sampled_static_obj_dict = ps_intention_sentence_list[0], robot_sampled_static_obj_dict_list[0]
        
            _, data_tmp_traits = create_data(ps_intention_sentence, None, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention", cls_type="traits")
            _, data_tmp_temporal = create_data(ps_intention_sentence, None, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention", cls_type="temporal")
            if i == 0 and j == 0:
                confidence_intention_traits = test_model(data_tmp_traits, None, pretrained=False)[0]
                torch.cuda.empty_cache()
                confidence_intention_temporal = test_model(data_tmp_temporal, None, pretrained=False)[0]
                torch.cuda.empty_cache()
            elif i > 0 and j == 0:
                confidence_intention_traits = test_model(data_tmp_traits, prev_qlora_traits_intention_dir, pretrained=True)[0]
                torch.cuda.empty_cache()
                confidence_intention_temporal = test_model(data_tmp_temporal, prev_qlora_temporal_intention_dir, pretrained=True)[0]
                torch.cuda.empty_cache()
            else:
                confidence_intention_traits = test_model(data_tmp_traits, qlora_traits_intention_dir, pretrained=True)[0]
                torch.cuda.empty_cache()
                confidence_intention_temporal = test_model(data_tmp_temporal, qlora_temporal_intention_dir, pretrained=True)[0]
                torch.cuda.empty_cache()
            
            confidence_intention = 0.7 * confidence_intention_traits + 0.3 * confidence_intention_temporal
            if confidence_intention > confidence_intention_threshold:
                selected_intention_sentence = ps_intention_sentence
                selected_sampled_static_obj_dict = robot_sampled_static_obj_dict
            else:
                selected_intention_sentence = gt_intention_sentence
                selected_sampled_static_obj_dict = human_sampled_static_obj_dict
            robot_intentions_hist.append(f"{time_}: {selected_intention_sentence}")
            
            # Robot Inferring Predicates
            robot_retrieved_predicates = retrieve_memory(f"Current time: {time_}. Intention: {selected_intention_sentence}", robot_predicates_hist, times, time_, predicates_num, DEVICE, decay_factor=0.95, top_k=5, retrieve_type="predicate")
            ps_robot_retrieved_predicates = retrieve_memory(f"Current time: {time_}. Intention: {ps_intention_sentence}", robot_predicates_hist, times, time_, predicates_num, DEVICE, decay_factor=0.95, top_k=5, retrieve_type="predicate")

            robot_conversation_hist = predicates_discovery_gpt4(data_path, i, scene_id, [j, time_], [selected_sampled_static_obj_dict, dynamic_obj_room_mapping], [robot_retrieved_intentions, robot_retrieved_predicates], inferred_traits, robot_conversation_hist, temperature_dict, model_dict, start_over=True)
            robot_thoughts, robot_acts = extract_thoughts_and_acts(robot_conversation_hist[-1][1])

            _, data_tmp_traits = create_data([robot_thoughts, extract_inhand_obj_robot(robot_acts)], [None]*predicates_num, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="predicates", cls_type="traits")
            _, data_tmp_temporal = create_data([robot_thoughts, extract_inhand_obj_robot(robot_acts)], [None]*predicates_num, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="predicates", cls_type="temporal")
            if i == 0 and j == 0:
                confidence_predicates_traits = test_model(data_tmp_traits, None, cls_type="traits", pretrained=False)
                torch.cuda.empty_cache()
                confidence_predicates_temporal = test_model(data_tmp_temporal, None, cls_type="temporal", pretrained=False)
                torch.cuda.empty_cache()
            elif i > 0 and j == 0:
                confidence_predicates_traits = test_model(data_tmp_traits, prev_qlora_traits_predicates_dir, cls_type="traits", pretrained=True)
                torch.cuda.empty_cache()
                confidence_predicates_temporal = test_model(data_tmp_temporal, prev_qlora_temporal_predicates_dir, cls_type="temporal", pretrained=True)
                torch.cuda.empty_cache()
            else:
                confidence_predicates_traits = test_model(data_tmp_traits, qlora_traits_predicates_dir, cls_type="traits", pretrained=True)
                torch.cuda.empty_cache()
                confidence_predicates_temporal = test_model(data_tmp_temporal, qlora_temporal_predicates_dir, cls_type="temporal", pretrained=True)
                torch.cuda.empty_cache()
            
            confidence_predicates = [(0.7 * x + 0.3 * y) for x, y in zip(confidence_predicates_traits, confidence_predicates_temporal)]
            confidence_intention_hist.append(confidence_intention)
            confidence_predicates_hist.extend(confidence_predicates)
            confidence_intention_avg = calculate_confidence_avg(confidence_intention_hist)
            confidence_predicates_avg = calculate_confidence_avg(confidence_predicates_hist)

            for k, (robot_thought, confidence_predicate, human_thought) in enumerate(zip(robot_thoughts, confidence_predicates, human_thoughts)):
                if confidence_predicate > confidence_predicate_threshold:
                    robot_predicates_hist.append(f"{time_}.{k}: {robot_thought}")
                else:
                    robot_predicates_hist.append(f"{time_}.{k}: {human_thought}")

            # Robot Inferring Human Traits
            inferred_traits = traits_inference_gpt4(data_path, i, scene_id, [j, time_], [robot_intentions_hist, robot_predicates_hist], inferred_traits, temperature_dict, model_dict, start_over=True)[0][1]
            inferred_traits = extract_scores(inferred_traits)
            inferred_traits_hist.append(inferred_traits)
            _, big_five_mse = calculate_ocean_mse(big_five, inferred_traits_hist)

            # Judge Approving Collaborations
            collaboration_approval = collaboration_approval_gpt4(data_path, i, scene_id, [j, time_], [gt_intention_sentence, ps_intention_sentence], human_thoughts, extract_inhand_obj_human(human_acts), robot_thoughts, robot_acts, temperature_dict, model_dict, start_over=True)[0][1]
            intention_approval, tasks_approval, _, _ = extract_collaboration(collaboration_approval)
            intention_approval_hist.append(intention_approval)
            tasks_approval_hist.extend(tasks_approval)
            intention_approval_score = calculate_accuracy(intention_approval_hist)
            tasks_approval_score = calculate_accuracy(tasks_approval_hist)

            # Finetune with QLoRA
            # Create intention training data
            if confidence_intention > confidence_intention_threshold:
                labels = 1 if intention_approval == "yes" else 0
                add_data_train_traits_intention, _ = create_data(selected_intention_sentence, labels, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention", cls_type="traits")
                add_data_train_temporal_intention, _ = create_data(selected_intention_sentence, labels, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention", cls_type="temporal")
            else:
                add_data_train_traits_intention, _ = create_data(selected_intention_sentence, 1, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention", cls_type="traits")
                add_data_train_temporal_intention, _ = create_data(selected_intention_sentence, 1, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention", cls_type="temporal")
                labels = 1 if intention_approval == "yes" else 0
                add_data_train_traits_intention_1,_ = create_data(ps_intention_sentence, labels, time_, inferred_traits, [robot_retrieved_intentions, ps_robot_retrieved_predicates], data_type="intention", cls_type="traits")
                add_data_train_temporal_intention_1, _ = create_data(ps_intention_sentence, labels, time_, inferred_traits, [robot_retrieved_intentions, ps_robot_retrieved_predicates], data_type="intention", cls_type="temporal")
                
                add_data_train_traits_intention.extend(add_data_train_traits_intention_1)
                add_data_train_temporal_intention.extend(add_data_train_temporal_intention_1)

            data_train_traits_intention.extend(add_data_train_traits_intention)
            data_train_temporal_intention.extend(add_data_train_temporal_intention)

            # Create predicates training data
            labels = [] 
            train_thoughts, train_acts, train_acts_1 = robot_thoughts, extract_inhand_obj_robot(robot_acts), []
            for k in range(predicates_num):
                label = 1 if tasks_approval[k] == "yes" else 0
                labels.append(label)

            for k, confidence_predicate in enumerate(confidence_predicates):
                if confidence_predicate <= confidence_predicate_threshold:
                    labels.append(1)
                    train_thoughts.append(human_thoughts[k])
                    train_acts_1.append(human_acts[k])
            if train_acts_1: train_acts.extend(extract_inhand_obj_human(train_acts_1))
            
            add_data_train_traits_predicates, _ = create_data([train_thoughts, train_acts], labels, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="predicates", cls_type="traits")
            add_data_train_temporal_predicates, _ = create_data([train_thoughts, train_acts], labels, time_, inferred_traits, [robot_retrieved_intentions, robot_retrieved_predicates], data_type="predicates", cls_type="temporal")
            data_train_traits_predicates.extend(add_data_train_traits_predicates)
            data_train_temporal_predicates.extend(add_data_train_temporal_predicates)

            # print()
            # print(12345, data_train_traits_intention)
            # print()
            # print(12345, data_train_temporal_intention)
            # print()
            # print(12345, data_train_traits_predicates)
            # print()
            # print(12345, data_train_temporal_predicates)
            # print()

            # ts = time.time()
            # time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            # qlora_traits_dir = os.path.join(data_path, f"qlora/traits/{str(i).zfill(5)}/{scene_id}/{time_string}_{time_}")
            # qlora_temporal_dir = os.path.join(data_path, f"qlora/temporal/{str(i).zfill(5)}/{scene_id}/{time_string}_{time_}")
            qlora_traits_intention_dir = os.path.join(data_path, f"qlora/traits/intention/{str(i).zfill(5)}/{scene_id}")
            qlora_temporal_intention_dir = os.path.join(data_path, f"qlora/temporal/intention/{str(i).zfill(5)}/{scene_id}")
            qlora_traits_predicates_dir = os.path.join(data_path, f"qlora/traits/predicates/{str(i).zfill(5)}/{scene_id}")
            qlora_temporal_predicates_dir = os.path.join(data_path, f"qlora/temporal/predicates/{str(i).zfill(5)}/{scene_id}")
            os.makedirs(qlora_traits_intention_dir, exist_ok=True)
            os.makedirs(qlora_temporal_intention_dir, exist_ok=True)
            os.makedirs(qlora_traits_predicates_dir, exist_ok=True)
            os.makedirs(qlora_temporal_predicates_dir, exist_ok=True)

            if i == 0:
                train_model(1, data_train_traits_intention, data_train_traits_intention, qlora_traits_intention_dir, checkpoint_dir=None, pretrained=False)
                torch.cuda.empty_cache()
                train_model(1, data_train_temporal_intention, data_train_temporal_intention, qlora_temporal_intention_dir, checkpoint_dir=None, pretrained=False)
                torch.cuda.empty_cache()
                train_model(1, data_train_traits_predicates, data_train_traits_predicates, qlora_traits_predicates_dir, checkpoint_dir=None, pretrained=False)
                torch.cuda.empty_cache()
                train_model(1, data_train_temporal_predicates, data_train_temporal_predicates, qlora_temporal_predicates_dir, checkpoint_dir=None, pretrained=False)
                torch.cuda.empty_cache()
            else:
                train_model(1, data_train_traits_intention, data_train_traits_intention, qlora_traits_intention_dir, checkpoint_dir=prev_qlora_traits_intention_dir, pretrained=True)
                torch.cuda.empty_cache()
                train_model(1, data_train_temporal_intention, data_train_temporal_intention, qlora_temporal_intention_dir, checkpoint_dir=prev_qlora_temporal_intention_dir, pretrained=True) 
                torch.cuda.empty_cache()
                train_model(1, data_train_traits_predicates, data_train_traits_predicates, qlora_traits_predicates_dir, checkpoint_dir=prev_qlora_traits_predicates_dir, pretrained=True)
                torch.cuda.empty_cache()
                train_model(1, data_train_temporal_predicates, data_train_temporal_predicates, qlora_temporal_predicates_dir, checkpoint_dir=prev_qlora_temporal_predicates_dir, pretrained=True) 
                torch.cuda.empty_cache()

            # Data Visualization
            for k in range(predicates_num):
                if k == 0:
                    csv_data.append([time_, gt_intention_sentence, ps_intention_sentence, confidence_intention, (confidence_intention_traits, confidence_intention_temporal), confidence_intention_avg, human_thoughts[k]+" "+ human_acts[k], robot_thoughts[k]+" "+ robot_acts[k], confidence_predicates[k], (confidence_predicates_traits[k], confidence_predicates_temporal[k]), confidence_predicates_avg, profile_string, big_five, inferred_traits, big_five_mse, intention_approval, intention_approval_score, tasks_approval, tasks_approval_score])
                else:
                    csv_data.append(["", "", "", "", "", "", human_thoughts[k]+" "+ human_acts[k], robot_thoughts[k]+" "+ robot_acts[k], confidence_predicates[k], (confidence_predicates_traits[k], confidence_predicates_temporal[k]), "", "", "", "", "", "", "", "", ""])
            csv_data.append(["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])


        header = ["Time", "Human Intention", "Robot Intention", "Robot Intention Confidence", "Robot Intention Confidence Breakdown", "Robot Intention Confidence Avg", "Human Predicate", "Robot Predicate", "Robot Predicate Confidence", "Robot Predicate Confidence Breakdown", "Robot Predicate Confidence Avg", "Human Traits", "Human Big 5", "Robot Big 5", "MSE", "Intention Approval", "Intention Approval Score", "Tasks Approval", "Tasks Approval Score"]
        df = pd.DataFrame(csv_data, columns=header)
        with pd.ExcelWriter(vis_csv_path, engine='openpyxl', mode='w') as writer: 
            df.to_excel(writer, index=False, sheet_name='vis')
            for column in df:
                column_length = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                writer.sheets['vis'].column_dimensions[get_column_letter(col_idx+1)].width = column_length + 2

        with open(vis_txt_path, 'w') as f:
            f.write("data_train_traits_intention:\n")
            f.write(str(data_train_traits_intention) + "\n\n")
            f.write("data_train_temporal_intention:\n")
            f.write(str(data_train_temporal_intention) + "\n\n")
            f.write("data_train_traits_predicates:\n")
            f.write(str(data_train_traits_predicates) + "\n\n")
            f.write("data_train_temporal_predicates:\n")
            f.write(str(data_train_temporal_predicates) + "\n\n")
