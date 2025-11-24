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
import argparse

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
os.chdir(dir_path)
sys.path.append(os.path.join(dir_path, "coopera_main"))

from sentence_transformers import SentenceTransformer

from habitat.gpt.prompts.utils import *
from skill_utils import *
from human_utils import *
from robot_utils import *
from judge_utils import *



def parse_args():
    parser = argparse.ArgumentParser(description='Human Simulation with Configurable Parameters')
    
    # Core configuration parameters
    parser.add_argument('--use-gpt-human', type=lambda x: x.lower() == 'true', 
                        default=True,
                        help='Read human simulation results by GPT, Llama if false (True/False, default: True)')
    
    parser.add_argument('--collab-type', type=int, default=2, choices=[1, 2],
                        help='Collaboration type (1 or 2, default: 2)')
    
    # Loop control parameters
    parser.add_argument('--max-days', type=int, default=5,
                        help='Maximum number of days to process (d > X, default: 5)')
    
    # Optional: specific indices to process
    parser.add_argument('--scene-indices', type=int, nargs='+', default=None,
                        help='Specific scene indices to process (e.g., --scene-indices 0 1 2)')
    parser.add_argument('--profile-indices', type=int, nargs='+', default=None,
                        help='Specific profile indices to process (e.g., --profile-indices 0 1 2)')
    parser.add_argument('--day-indices', type=int, nargs='+', default=None,
                        help='Specific day indices to process (e.g., --day-indices 0 1 2)')
    parser.add_argument('--hour-indices', type=int, nargs='+', default=None,
                        help='Specific hour indices to process (e.g., --hour-indices 0 1 2)')
    
    # GPU configuration
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='the GPU id to run Habitat simulation and put Sentence Transformer')
    
    return parser.parse_args()



os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parse_args()
DEVICE = f"cuda:{args.gpu_id}"
set_seed_everywhere(42)

# Load model for semantic similarity
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)



# conda activate /hdd2/kai/Dynamic_Human_Robot_Value_Alignments/env
# CUDA_VISIBLE_DEVICES="0" python coopera_main/human_sim/human_sim_render.py --use-gpt-human True --collab-type 2
# watch -n 1 nvidia-smi
if __name__ == "__main__":
    data_path = os.path.join(dir_path, "data")
    results_path = os.path.join(dir_path, "results")
    replay_dir = os.path.join(results_path, "replays")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

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
    npy_file_folder_list = [os.path.join(data_path, "humanoids/humanoid_data/all_motion")]  # Combined here
                            
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

    # Select scenes
    scene_id_list = ["103997919_171031233", "108736635_177263256", "105515211_173104179", "108736872_177263607", "102344049"]

    # GPT or Llama
    use_gpt_human = args.use_gpt_human

    # Collaboration type
    collab_type = args.collab_type

    # List of avaiablie GPT models (https://platform.openai.com/docs/pricing)
    model_dict = {
        "traits_summary": "gpt-5.1",
        "intention_proposal": "gpt-5.1",
        "predicates_proposal": "gpt-5.1",
        "predicates_reflection": "gpt-5.1",
        "intention_discovery": "gpt-5.1",
        "predicates_discovery": "gpt-5.1",
        "traits_inference": "gpt-5.1",
        "collaboration_approval": "gpt-5.1"
    }

    # Set temperature fo LLMs
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

    # Override if using GPT-5-series models
    if any("gpt-5" in model for model in model_dict.values()) and use_gpt_human:
        temperature_dict = {temp: 1 for temp in temperature_dict}

    profile_string_list, big_five_list = read_human_data_mypersonality(data_path)
    times = ['9 am', '10 am', '11 am', '12 pm', 
            '1 pm', '2 pm', '3 pm', '4 pm', 
            '5 pm', '6 pm', '7 pm', '8 pm', '9 pm'
    ]
    if collab_type == 1:
        predicates_num, intentions_num = 3, 3
    elif collab_type == 2:
        predicates_num, intentions_num = 5, 5
    days = [str(i).zfill(2) for i in range(args.max_days)]


    for sd, scene_id in enumerate(scene_id_list):
        if args.scene_indices is not None:
            if sd not in args.scene_indices:
                continue

        agent_dict = {"agent_0": robot_agent_config, "agent_1": human_agent_config}
        env = create_agent_action(agent_dict, scene_id, collab_type=collab_type)
        env.reset()
        
        room_dict, static_obj_trans_dict, dynamic_obj_trans_dict, static_obj_room_mapping, dynamic_obj_room_mapping, aom, rom = select_pick_place_obj(env, scene_id, 0, 0, collab_type=collab_type)
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

        for i, (profile_string, big_five) in enumerate(zip(profile_string_list, big_five_list)):
            if args.profile_indices is not None:
                if i not in args.profile_indices:
                    continue
            
            # profile_string = traits_summary_mllm(results_path, i, scene_id, [profile_string, big_five], temperature_dict, model_dict, gpt=use_gpt_human, start_over=False)[0][1]
            # human_intentions_hist, human_predicates_hist = [], []

            for d, day in enumerate(days):
                if args.day_indices is not None:
                    if d not in args.day_indices:
                        continue

                for j, time_ in enumerate(times):
                    if args.hour_indices is not None:
                        if j not in args.hour_indices:
                            continue

                    # # Human Proposing Intention
                    # human_retrieved_intentions = retrieve_memory(f"Current day: {day}. Current time: {time_}", human_intentions_hist, d, times, time_, predicates_num, sentence_model, decay_factor=0.95, top_k=13, retrieve_type="intention")
                    # human_retrieved_predicates = retrieve_memory(f"Current day: {day}. Current time: {time_}", human_predicates_hist, d, times, time_, predicates_num, sentence_model, decay_factor=0.95, top_k=10, retrieve_type="predicate")

                    # human_conversation_hist = intention_proposal_mllm(results_path, i, scene_id, [day, j, time_], [human_retrieved_intentions, human_retrieved_predicates], room_list, [profile_string, big_five], temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)
                    # _, gt_intention_sentence_list, human_sampled_static_obj_dict_list = sample_obj_by_similarity(human_conversation_hist, static_obj_room_mapping, sentence_model, top_k=30)
                    # _, sampled_motion_list = sample_motion_by_similarity(human_conversation_hist, motion_sets_list, sentence_model, top_k=5)
                    # gt_intention_sentence, human_sampled_static_obj_dict, sampled_motion_list = gt_intention_sentence_list[0], human_sampled_static_obj_dict_list[0], sampled_motion_list[0]
                    # human_intentions_hist.append(f"day {d} time {time_}: {gt_intention_sentence}")

                    # # Human Proposing Predicates
                    # human_retrieved_predicates = retrieve_memory(f"Current day: {day}. Current time: {time_}. Intention: {gt_intention_sentence}", human_predicates_hist, d, times, time_, predicates_num, sentence_model, decay_factor=0.95, top_k=10, retrieve_type="predicate")

                    # human_conversation_hist = predicates_proposal_mllm(results_path, i, scene_id, [day, j, time_], gt_intention_sentence, [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)
                    # human_conversation_hist = predicates_reflection_1_mllm(results_path, i, scene_id, [day, j, time_], gt_intention_sentence, [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)
                    # human_conversation_hist = predicates_reflection_2_mllm(results_path, i, scene_id, [day, j, time_], gt_intention_sentence, [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)

                    # human_thoughts, human_acts = extract_thoughts_and_acts(human_conversation_hist[-1][1], search_txt=" Reason_human:")
                    # if not human_thoughts: human_thoughts, human_acts = extract_thoughts_and_acts(human_conversation_hist[-1][1], search_txt="")
                    # human_predicates_hist.extend([f"day {d} time {time_} task {k}: {human_thought}" for k, human_thought in enumerate(human_thoughts)])


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
                    
                    if use_gpt_human:
                        extracted_planning = extract_code("predicates_reflection_2", pathlib.Path(results_path) / "human/gpt_response" / f"collaboration_{collab_type}/predicates_reflection_2" / str(i).zfill(5) / scene_id / day, j, collab=collab_type)
                    else:
                        extracted_planning = extract_code("predicates_reflection_2", pathlib.Path(results_path) / "human/llama_response" / f"collaboration_{collab_type}/predicates_reflection_2" / str(i).zfill(5) / scene_id / day, j, collab=collab_type)
                    
                    if collab_type == 1:
                        execute_humanoid_1(env, observations, replay_dir, humanoid_rearrange_controller, i, scene_id, day, time_, extracted_planning, motion_sets_list, [static_obj_room_mapping, dynamic_obj_room_mapping], [static_obj_trans_dict, dynamic_obj_trans_dict], room_dict, sentence_model, gpt=use_gpt_human)
                    elif collab_type == 2:
                        execute_humanoid_2(env, observations, convert_helper, folder_dict, motion_dict, human_urdf_path, replay_dir, humanoid_rearrange_controller, i, scene_id, day, time_, extracted_planning, motion_sets_list, npy_file_folder_list, [static_obj_room_mapping, dynamic_obj_room_mapping], [static_obj_trans_dict, dynamic_obj_trans_dict], room_dict, sentence_model, gpt=use_gpt_human)

        # End the current environment of the scene instance
        env.close()
