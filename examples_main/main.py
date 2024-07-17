import habitat_sim
import magnum as mn
import warnings
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
import copy
import random
import torch
import pathlib
import time

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)

from sentence_transformers import SentenceTransformer

from habitat.gpt.prompts.utils import *
from human_utils import *
from robot_utils import *


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
        os.path.join(output_dir, "robot_scene_camera_rgb_video.mp4"),
        open_vid=False,  # Ensure this is set to False to prevent video from popping up
    )
    vut.make_video(
        observations,
        "agent_0_third_rgb",
        "color",
        os.path.join(output_dir, "robot_third_rgb_video.mp4"),
        open_vid=False,
    )
    vut.make_video(
        observations,
        "agent_1_head_rgb",
        "color",
        os.path.join(output_dir, "human_scene_camera_rgb_video.mp4"),
        open_vid=False, 
    )
    vut.make_video(
        observations,
        "agent_1_third_rgb",
        "color",
        os.path.join(output_dir, "human_third_rgb_video.mp4"),
        open_vid=False,
    )
    vut.make_video(
        observations,
        "agent_1_top_rgb",
        "color",
        os.path.join(output_dir, "top_scene_camera_rgb_video.mp4"),
        open_vid=False,  # Ensure this is set to False to prevent video from popping up
    )


def convert_npy_to_npz(npy_file_path, output_npz_path):
    """
    Convert motion data from a .npy file into separate motion parameters and save them as .npy and .npz files.

    Parameters:
    npy_file_path (str): Path to the input .npy file containing motion data.
    output_npz_path (str): Path to the output .npz file.
    trans_order (str): Order of the translation coordinates. Default is 'xyz'.
    trans_signs (str): Signs for the translation coordinates. Use '+' for positive and '-' for negative.
    """
    # Load the provided .npy file
    motion = np.load(npy_file_path, allow_pickle=True)

    # Convert to PyTorch tensor and float
    motion = torch.tensor(motion).float()

    # Extract motion parameters
    # https://github.com/IDEA-Research/Motion-X
    motion_parms = {
        'root_orient': motion[:, :3].numpy(),  # controls the global root orientation
        'pose_body': motion[:, 3:3+63].numpy(),  # controls the body
        'pose_hand': motion[:, 66:66+90].numpy(),  # controls the finger articulation
        'pose_jaw': motion[:, 66+90:66+93].numpy(),  # controls the yaw pose
        'face_expr': motion[:, 159:159+50].numpy(),  # controls the face expression
        'face_shape': motion[:, 209:209+100].numpy(),  # controls the face shape
        'trans': motion[:, 309:309+3].numpy(),  # controls the global body position
        'betas': motion[:, 312:].numpy(),  # controls the body shape. Body shape is static
    }

    trans_order='xyz'
    trans_signs='+++'
    # Reorder the translation coordinates based on the given order
    order_map = {'x': 0, 'y': 1, 'z': 2}
    trans_order_indices = [order_map[axis] for axis in trans_order]
    motion_parms['trans'] = motion_parms['trans'][:, trans_order_indices]

    # Apply signs to the translation coordinates
    for i, sign in enumerate(trans_signs):
        if sign == '-':
            motion_parms['trans'][:, i] *= -1

    orient_order='xyz'
    orient_signs='+++'
    # Reorder the orientation coordinates based on the given order
    orient_order_indices = [order_map[axis] for axis in orient_order]
    motion_parms['root_orient'] = motion_parms['root_orient'][:, orient_order_indices]

    # Apply signs to the orientation coordinates
    for i, sign in enumerate(orient_signs):
        if sign == '-':
            motion_parms['root_orient'][:, i] *= -1

    # Create the output directory based on the .npz file name
    output_dir = os.path.splitext(output_npz_path)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Save each parameter as a .npy file
    for key, value in motion_parms.items():
        np.save(os.path.join(output_dir, f'{key}.npy'), value)

    # Create poses.npy with a dimension of 165
    num_frames = motion.shape[0]
    zero_vec = np.zeros((num_frames, 3))
    poses = np.concatenate([
        zero_vec,
        motion_parms['pose_body'],  # kind of certain
        motion_parms['pose_jaw'],
        motion_parms['root_orient'],
        motion_parms['trans'],
        motion_parms['pose_hand'],  # kind of certain
    ], axis=1)
    
    np.save(os.path.join(output_dir, 'poses.npy'), poses)
    motion_parms['poses'] = poses

    # Create an .npz file containing all the parameters
    np.savez(output_npz_path, **motion_parms)

    # Verify the saved .npz file
    saved_npz = np.load(output_npz_path)
    saved_npz_keys = saved_npz.files
    saved_npz_shapes = {key: saved_npz[key].shape for key in saved_npz_keys}

    return saved_npz_keys, saved_npz_shapes


def create_motion_sets(npy_file_folder_list, human_urdf_path, update=True):
    # motion_npz_path = os.path.join(data_path, "humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.npz")
    # motion_pkl_path = os.path.join(data_path, "humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.pkl")
    motion_sets_list, motion_pkl_path_list, folder_list = [], [], []
    convert_helper = MotionConverterSMPLX(urdf_path=human_urdf_path)

    for folder in npy_file_folder_list:
        # Ensure the folder exists
        if not os.path.isdir(folder):
            print(f"Folder does not exist: {folder}")
            continue
        
        # Iterate through each file in the folder
        for filename in os.listdir(folder):
            if filename.endswith(".npy"):
                # Full path of the .npy file
                npy_full_path = os.path.join(folder, filename)

                # Extract the relative path starting from 'data/'
                motion_npy_path = os.path.join("data", npy_full_path.split('data/', 1)[1])
                
                # Create the base name for .npz and .pkl by removing the .npy extension
                base_name = filename[:-4]
                
                # Create the corresponding .npz and .pkl paths
                motion_npz_path = motion_npy_path.replace('.npy', '.npz')
                motion_pkl_path = motion_npy_path.replace('.npy', '.pkl')

                if update:
                    convert_npy_to_npz(motion_npy_path , motion_npz_path)
                    convert_helper.convert_motion_file(
                        motion_path=motion_npz_path,
                        output_path=motion_npz_path.replace(".npz", ""),
                    )
                motion_sets_list.append(base_name)
                motion_pkl_path_list.append(motion_pkl_path)
                folder_list.append(folder)

    motion_dict = dict(zip(motion_sets_list, motion_pkl_path_list))
    folder_dict = dict(zip(motion_pkl_path_list, folder_list))
    return motion_sets_list, motion_dict, folder_dict, convert_helper


def get_motion_pkl_path(motion_set_name, motion_dict):
    return motion_dict.get(motion_set_name, None)


def interpolate_points(start, end, steps):
    return [start + (end - start) * (i / steps) for i in range(1, steps + 1)]


def set_agents_base_pos(env, robot_pos, human_pos):
    env.sim.agents_mgr[0].articulated_agent.base_pos = robot_pos
    env.sim.agents_mgr[1].articulated_agent.base_pos = human_pos


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


def create_static_obj_trans_dict(instance_file, object_mapping, static_categories):
    """
    Create a dictionary with actual object names and their translations, only if they belong to specified static categories.
    """
    object_translation_dict = {}
    # TODO: multiple same objects will be added as a single instance
    i = 0
    for obj in instance_file['object_instances']:
        template_name = obj['template_name']
        if template_name in object_mapping:
            actual_name = object_mapping[template_name]['name']
            if pd.isna(actual_name): actual_name = object_mapping[template_name]['wnsynsetkey']
            super_category = object_mapping[template_name]['super_category']
            # if super_category in static_categories:
            if True:
                translation = obj['translation']
                object_translation_dict[actual_name] = [i, mn.Vector3(translation)]
                i += 1
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

    def find_closest_room(object_translation, room_bounds_mapping):
        """
        Find the closest room to the object based on its translation.
        """
        closest_room = None
        min_distance = float('inf')

        for room, bounds in room_bounds_mapping.items():
            min_bounds = bounds["min_bounds"]
            max_bounds = bounds["max_bounds"]
            distance = 0

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

        return closest_room
    
    obj_room_mapping = {}
    for obj_name, translation in object_trans_dict.items():
        room = get_room_for_object(translation[1], room_bounds_mapping)
        if room:
            obj_room_mapping[obj_name] = [translation[0], room]
        else:
            closest_room = find_closest_room(translation[1], room_bounds_mapping)
            if closest_room:
                obj_room_mapping[obj_name] = [translation[0], closest_room]
    
    return obj_room_mapping


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
    with open(instance_file, 'r') as f: instance_data = json.load(f)
    static_obj_trans_dict = create_static_obj_trans_dict(instance_data, obj_mapping, static_categories)
    dynamic_obj_trans_dict = {}
    
    aom = env.sim.get_articulated_object_manager()
    rom = env.sim.get_rigid_object_manager()

    # We can query the articulated and rigid objects
    print("\nList of dynamic articulated objects:")
    for handle, ao in aom.get_objects_by_handle_substring().items():
        print(handle, "id", aom.get_object_id_by_handle(handle))

    print("\nList of dynamic rigid objects:")
    for handle, ro in rom.get_objects_by_handle_substring().items():
        if ro.awake:
            print(handle, "id", ro.object_id)
            template_name = handle.split('_:')[0]
            trans = (rom.get_object_by_id(ro.object_id)).translation
            actual_name = obj_mapping[template_name]['name'] if template_name in obj_mapping else handle
            dynamic_obj_trans_dict[actual_name] = [ro.object_id, trans]
    
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


def pick_up(env, humanoid_controller, pick_obj_id, pick_object_trans):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)  # This line is important
    for _ in range(100):
        action_dict = {"action": ("agent_1_humanoid_pick_obj_id_action"), "action_args": {"agent_1_humanoid_pick_obj_id": pick_obj_id}}
        observations.append(env.step(action_dict)) 


def walk_to(env, humanoid_controller, place_object_trans):
    # https://github.com/facebookresearch/habitat-lab/issues/1913
    humanoid_controller.reset(env.sim.agents_mgr[1].articulated_agent.base_transformation)  # This line is important

    # Walk towards the object to place
    agent_displ = np.inf
    agent_rot = np.inf
    prev_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
    prev_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
    while agent_displ > 10 or agent_rot > 1e-1:  # TODO: change from threshold of 1e-9 to 1e-3 avoids the OOM issue
        prev_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
        prev_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        action_dict = {
            "action": ("agent_1_humanoid_navigate_action", "agent_0_oracle_coord_action"),  
            "action_args": {
                "agent_1_oracle_nav_lookat_action": place_object_trans,
                "agent_1_mode": 1,
                "agent_0_oracle_nav_lookat_action": prev_pos,
                "agent_0_mode": 1
            }
        }
        observations.append(env.step(action_dict))
        
        cur_rot = env.sim.agents_mgr[1].articulated_agent.base_rot
        cur_pos = env.sim.agents_mgr[1].articulated_agent.base_pos
        agent_displ = (cur_pos - place_object_trans).length()  # agent_displ = (cur_pos - prev_pos).length()
        agent_rot = np.abs(cur_rot - prev_rot)

    # Wait
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))


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


def execute_humanoid_1(env, extracted_planning, motion_sets_list, obj_room_mapping, obj_trans_dict):
    # TODO: Using sudo dmesg -T, the process is sometimes killed because OOM Killer. 
    # The reason is likely to be for some free-form motion, the robot is in collision with the scene, and increases computation overhead.
    # When rendering the videos, it causes OOM Killer.
    static_obj_room_mapping, dynamic_obj_room_mapping = obj_room_mapping[0], obj_room_mapping[1]
    static_obj_trans_dict, dynamic_obj_trans_dict = obj_trans_dict[0], obj_trans_dict[1]
    planning = extracted_planning[list(extracted_planning.keys())[0]]["Predicates"]

    for i, step in enumerate(planning):  # planning for each predicate
        if i != 0: continue
        obj_trans_dict_to_search = static_obj_trans_dict  # only dynamic object can be picked or placed
        object_trans = None
        for name, (obj_id, trans) in obj_trans_dict_to_search.items():
            if obj_id == step[1]:
                object_trans = trans
                break
        
        walk_to(env, humanoid_rearrange_controller, object_trans)
        
        if step[0] == 1:
            selected_motion = most_similar_motion(step[4], motion_sets_list)[0]
            customized_humanoid_motion(env, convert_helper, folder_dict, get_motion_pkl_path(selected_motion, motion_dict))
            print()
            print(selected_motion)
            print()
        else:
            if step[0] == 2:
                pick_up(env, humanoid_rearrange_controller, step[1], object_trans)
            elif step[0] == 3:
                move_hand_and_place(env, humanoid_rearrange_controller, step[1], object_trans)
        print("step done")
   
        make_videos(output_dir)
        extract_frames(os.path.join(output_dir, "human_third_rgb_video.mp4"), os.path.join(output_dir, "human_third_rgb_video"))
    
    # make_videos(output_dir)


def read_human_data():
    csv_file_path = os.path.join(data_path, "humanoids/humanoid_data/okcupid_profiles.csv")
    okcupid_data = pd.read_csv(csv_file_path)
    profiles = okcupid_data[['age', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'education', 'ethnicity', 'height', 'job', 'location', 'offspring', 'pets', 'religion', 'smokes', 'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6','essay7', 'essay8', 'essay9']]

    profile_string_complete_list = []
    profile_string_partial_list = []

    for i in range(min(100, len(profiles))):
        selected_profile = profiles.iloc[i]

        # Create a one-paragraph string
        profile_string_complete = (
            f"Age: {selected_profile['age']}; "
            f"Sex: {selected_profile['sex']}; "
            f"Orientation: {selected_profile['orientation']}; "
            f"Body Type: {selected_profile['body_type']}; "
            f"Diet: {selected_profile['diet']}; "
            f"Drinks: {selected_profile['drinks']}; "
            f"Education: {selected_profile['education']}; "
            f"Ethnicity: {selected_profile['ethnicity']}; "
            f"Height: {selected_profile['height']}; "
            f"Job: {selected_profile['job']}; "
            f"Location: {selected_profile['location']}; "
            f"Offspring: {selected_profile['offspring']}; "
            f"Pets: {selected_profile['pets']}; "
            f"Religion: {selected_profile['religion']}; "
            f"Smokes: {selected_profile['smokes']}; "
            f"Intro 1: {selected_profile['essay0']}; "
            f"Intro 2: {selected_profile['essay1']}; "
            f"Intro 3: {selected_profile['essay2']}; "
            f"Intro 4: {selected_profile['essay3']}; "
            f"Intro 5: {selected_profile['essay4']}; "
            f"Intro 6: {selected_profile['essay5']}; "
            f"Intro 7: {selected_profile['essay6']}; "
            f"Intro 8: {selected_profile['essay7']}; "
            f"Intro 9: {selected_profile['essay8']}; "    
            f"Intro 10: {selected_profile['essay9']}"  
        )

        profile_string_partial = (
            f"Age: {selected_profile['age']}; "
            f"Sex: {selected_profile['sex']}; "
            f"Orientation: {selected_profile['orientation']}; "
            f"Body Type: {selected_profile['body_type']}; "
            f"Diet: {selected_profile['diet']}; "
            f"Drinks: {selected_profile['drinks']}; "
            f"Education: {selected_profile['education']}; "
            f"Ethnicity: {selected_profile['ethnicity']}; "
            f"Height: {selected_profile['height']}; "
            f"Job: {selected_profile['job']}; "
            f"Location: {selected_profile['location']}; "
            f"Offspring: {selected_profile['offspring']}; "
            f"Pets: {selected_profile['pets']}; "
            f"Religion: {selected_profile['religion']}; "
            f"Smokes: {selected_profile['smokes']}."
        )

        profile_string_complete_list.append(profile_string_complete)
        profile_string_partial_list.append(profile_string_partial)

    return profile_string_complete_list, profile_string_partial_list



os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if __name__ == "__main__":
    output_dir = os.path.join(data_path, "interactive_play_replays")
    os.makedirs(output_dir, exist_ok=True)

    # Define the robot agent configuration
    robot_agent_config = AgentConfig()
    urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
    robot_agent_config.articulated_agent_urdf = urdf_path
    robot_agent_config.articulated_agent_type = "FetchRobot"

    # Define the human agent configuration
    human_agent_config = AgentConfig()
    urdf_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0.urdf")
    human_agent_config.articulated_agent_urdf = urdf_path
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
                            
    motion_sets_list, motion_dict, folder_dict, convert_helper = create_motion_sets(npy_file_folder_list, urdf_path, update=True)
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
    # set_agents_base_pos(env, mn.Vector3(0, 0.180179, 0), mn.Vector3(0, 0.180179, 0))
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
      "intention_proposal": 0.9,
      "predicates_proposal": 0.9,
      "predicates_reflection": 0.25,
      "intention_discovery": 0.9,
      "predicates_discovery": 0.9,
      "collaboration_proposal": 0.7
    }
    # GPT-4 1106-preview is GPT-4 Turbo (https://openai.com/pricing)
    model_dict = {
      "intention_proposal": "gpt-4o",
      "predicates_proposal": "gpt-4o",
      "predicates_reflection": "gpt-4o",
      "intention_discovery": "gpt-4o",
      "predicates_discovery": "gpt-4o",
      "collaboration_proposal": "gpt-4o"
    }

    profile_string_complete_list, profile_string_partial_list = read_human_data()
    
    for i, profile_string_complete in enumerate(profile_string_complete_list):
        profile_string_partial = profile_string_partial_list[i]
        if i != 0: continue

        human_conversation_hist = intention_proposal_gpt4(data_path, i, scene_id, room_list, profile_string_complete, temperature_dict, model_dict, start_over=False)
        times, intention_sentences, sampled_static_obj_dict_list = sample_obj_by_similarity(human_conversation_hist, static_obj_room_mapping, top_k=30)
        _, sampled_motion_list = sample_motion_by_similarity(human_conversation_hist, motion_sets_list, top_k=5)


        human_conversation_hist = predicates_proposal_gpt4(data_path, i, scene_id, times, sampled_motion_list, sampled_static_obj_dict_list, dynamic_obj_room_mapping, profile_string_partial, human_conversation_hist, temperature_dict, model_dict, start_over=False)
        human_conversation_hist = predicates_reflection_gpt4(data_path, i, scene_id, times, sampled_motion_list, sampled_static_obj_dict_list, dynamic_obj_room_mapping, profile_string_partial, human_conversation_hist, temperature_dict, model_dict, start_over=False)

        selected_time = "9 am"
        for i, time_ in enumerate(times):
            if time_ == selected_time:
                break
        
        extracted_planning = extract_code("predicates_reflection", pathlib.Path(data_path) / "gpt4_response" / "human/predicates_reflection" / scene_id / str(i).zfill(5), i)

        execute_humanoid_1(env, extracted_planning, motion_sets_list, [static_obj_room_mapping, dynamic_obj_room_mapping], [static_obj_trans_dict, dynamic_obj_trans_dict])
        
        # robot_conversation_hist = intention_discovery_gpt4(data_path, scene_id, time_, os.path.join(output_dir, "human_third_rgb_video"), temperature_dict, model_dict, start_over=False)
        # # print(extracted_planning[f"Time: {selected_time}"]["Intention"])
        # # if extract_confidence(robot_conversation_hist[0][1]) == "yes" or extract_confidence(robot_conversation_hist[0][1]) == "Yes":
        
        # robot_conversation_hist = predicates_discovery_gpt4(data_path, scene_id, time_, robot_conversation_hist, temperature_dict, model_dict, start_over=False)
        # thought, act = extract_thoughts_and_acts(robot_conversation_hist[1][1])
        # predicate = extract_predicates(robot_conversation_hist[0][1])[0]
        # human_conversation_hist = collaboration_proposal_gpt4(data_path, scene_id, time_, sampled_motion_list[3], extracted_planning, predicate, thought, act, human_conversation_hist, temperature_dict, model_dict, start_over=False)