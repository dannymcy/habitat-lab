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
    parser.add_argument('--collab-type', type=int, default=2, choices=[1, 2],
                        help='Collaboration type (1 or 2, default: 2)')

    parser.add_argument('--collab-setting', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Collaboration type (1/2/3/4, default: 1)')

    parser.add_argument('--use-gpt-human', type=lambda x: x.lower() == 'true', 
                        default=True,
                        help='Read human simulation results by GPT, Llama if false (True/False, default: True)')

    parser.add_argument('--use-gpt-robot', type=lambda x: x.lower() == 'true', 
                        default=True,
                        help='Use GPT for robot intention/task discovery (True/False, default: True)')

    parser.add_argument('--start-logic-robot', type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Restart robot intention/task discovery (True/False, default: True)')

    parser.add_argument('--start-logic-lora', type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Restart robot intention/task classification (True/False, default: True)')
    
    # GPU configuration
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='the GPU id to run Habitat simulation and put Sentence Transformer')
    
    return parser.parse_args()


# Configuration for different collaboration settings
COLLABORATION_CONFIGS = {
    1: {
        'name': 'Same human, same scene',
        'scenes': [0] * 5,  # Same scene for 5 days
        'humans': [0] * 5,  # Same human for 5 days  
        'days': list(range(5)),
        'human_num': 1,
        'day_num_per_human': 5
    },
    2: {
        'name': 'Same human, different scenes',
        'scenes': [0, 1, 2, 3, 4],  # 5 different scenes
        'humans': [0] * 5,  # Same human
        'days': list(range(5)),
        'human_num': 1,
        'day_num_per_human': 5
    },
    3: {
        'name': 'Different humans, same scene',
        'scenes': [0] * 9,  # Same scene
        'humans': [0, 1, 2] * 3,  # Rotate 3 humans, 3 times
        'days': [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 9 days total
        'human_num': 3,
        'day_num_per_human': 3
    },
    4: {
        'name': 'Different humans, different scenes',
        'scenes': [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 3 scenes
        'humans': [0, 1, 2] * 3,  # Rotate humans in each scene
        'days': [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 9 days total
        'human_num': 3,
        'day_num_per_human': 3
    }
}


os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parse_args()
DEVICE = f"cuda:{args.gpu_id}"
set_seed_everywhere(42)

# Load model for semantic similarity
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)



# conda activate /hdd2/kai/Dynamic_Human_Robot_Value_Alignments/env
# CUDA_VISIBLE_DEVICES="0,2,3" python coopera_main/benchmark/main.py --collab-type 2 --collab-setting 2 --start-logic-robot False --start-logic-lora False
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

    # Collaboration type
    collab_type = args.collab_type

    # Collaboration setting
    collab_setting = args.collab_setting

    # GPT or Llama for human
    use_gpt_human = args.use_gpt_human

    # GPT or Llama for robot
    use_gpt_robot = args.use_gpt_robot

    # Restart robot intention/task discovery or not
    start_logic_robot = args.start_logic_robot

    # Restart robot intention/task classification or not
    start_logic_lora = args.start_logic_lora

    # List of avaiablie GPT models (https://platform.openai.com/docs/pricing)
    model_dict = {
        "traits_summary": "gpt-5.1",
        "intention_proposal": "gpt-5.1",
        "predicates_proposal": "gpt-5.1",
        "predicates_reflection": "gpt-5.1",
        "intention_discovery": "gpt-5.1",
        "predicates_discovery": "gpt-5.1",
        "traits_inference": "gpt-5.1",
        "collaboration_approval": "gpt-5.1",
        "open_ai_text_embedding": "text-embedding-ada-002"
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

    config = COLLABORATION_CONFIGS[collab_setting]
    days = [str(i).zfill(2) for i in range(config['day_num_per_human'])]

    # Initialize histories with dictionary comprehension
    inferred_traits_hist = {i: [] for i in range(config['human_num'])}
    inferred_profile_hist = {i: [] for i in range(config['human_num'])}
    human_intentions_hist = {i: [] for i in range(config['human_num'])}
    human_predicates_hist = {i: [] for i in range(config['human_num'])}
    robot_intentions_hist = {i: [] for i in range(config['human_num'])}
    robot_predicates_hist = {i: [] for i in range(config['human_num'])}
    inferred_traits, inferred_profile = [""] * config['human_num'], [""] * config['human_num']

    # Other initializations
    lora_intention_dir, lora_predicates_dir = "", ""
    data_train_intentions, data_train_predicates = [], []
    eval_big_five_across_days_latest, eval_big_five_across_days_voting = "", ""
    eval_intentions_llm_across_days, eval_predicates_llm_across_days, eval_predicates_semantic_across_days = [], [], []


    day_counter = 0
    for day_idx, scene_idx, human_idx in zip(config['days'], config['scenes'], config['humans']):
        file_idx = 0        
        day = days[day_idx]
        scene_id = scene_id_list[scene_idx]
        profile_string = profile_string_list[human_idx]
        big_five = big_five_list[human_idx]

        print(f"\033[93m\n=== Day {day_counter}, Scene {scene_idx} ({scene_id}), Human {human_idx}, Day {day_idx} for this human ===\033[0m")

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

        profile_string = traits_summary_mllm(results_path, human_idx, scene_id, [profile_string, big_five], temperature_dict, model_dict, gpt=use_gpt_human, start_over=False)[0][1]

        # Setup evaluation paths for this iteration
        eval_dir = pathlib.Path(results_path) / "benchmark" / f"collaboration_{collab_type}" / f"setting_{collab_setting}" / "main" / str(human_idx).zfill(5) / scene_id
        eval_csv_path = eval_dir / "eval.xlsx"
        eval_txt_path = eval_dir / f"eval_day_{day}.txt"
        eval_json_path = eval_dir / "eval.json"
        eval_csv_data = []
        os.makedirs(eval_dir, exist_ok=True)

        answer_intentions_within_day, answer_predicates_within_day = [], []
        labels_intentions_llm_within_day, labels_predicates_llm_within_day, labels_predicates_category_within_day = [], [], []
        semantic_sim = []

        prev_lora_intention_dir = lora_intention_dir
        prev_lora_predicates_dir = lora_predicates_dir

        for j, time_ in enumerate(times):
            # Human Proposing Intention
            human_retrieved_intentions = retrieve_memory(f"Current day: {day}. Current time: {time_}", human_intentions_hist[human_idx], day_idx, times, time_, predicates_num, sentence_model, decay_factor=0.95, top_k=13, retrieve_type="intention")
            human_retrieved_predicates = retrieve_memory(f"Current day: {day}. Current time: {time_}", human_predicates_hist[human_idx], day_idx, times, time_, predicates_num, sentence_model, decay_factor=0.95, top_k=10, retrieve_type="predicate")

            human_conversation_hist = intention_proposal_mllm(results_path, human_idx, scene_id, [day, j, time_], [human_retrieved_intentions, human_retrieved_predicates], room_list, [profile_string, big_five], temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)
            _, gt_intention_sentence_list, human_sampled_static_obj_dict_list = sample_obj_by_similarity(human_conversation_hist, static_obj_room_mapping, sentence_model, top_k=30)
            _, sampled_motion_list = sample_motion_by_similarity(human_conversation_hist, motion_sets_list, sentence_model, top_k=5)
            gt_intention_sentence, human_sampled_static_obj_dict, sampled_motion_list = gt_intention_sentence_list[0], human_sampled_static_obj_dict_list[0], sampled_motion_list[0]
            human_intentions_hist[human_idx].append(f"day {day_idx} time {time_}: {gt_intention_sentence}")

            # Human Proposing Predicates
            human_retrieved_predicates = retrieve_memory(f"Current day: {day}. Current time: {time_}. Intention: {gt_intention_sentence}", human_predicates_hist[human_idx], day_idx, times, time_, predicates_num, sentence_model, decay_factor=0.95, top_k=10, retrieve_type="predicate")

            human_conversation_hist = predicates_proposal_mllm(results_path, human_idx, scene_id, [day, j, time_], gt_intention_sentence, [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)
            human_conversation_hist = predicates_reflection_1_mllm(results_path, human_idx, scene_id, [day, j, time_], gt_intention_sentence, [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)
            human_conversation_hist = predicates_reflection_2_mllm(results_path, human_idx, scene_id, [day, j, time_], gt_intention_sentence, [human_retrieved_intentions, human_retrieved_predicates], sampled_motion_list, [human_sampled_static_obj_dict, dynamic_obj_room_mapping], [profile_string, big_five], human_conversation_hist, temperature_dict, model_dict, gpt=use_gpt_human, collab=collab_type, start_over=False)

            human_thoughts, human_acts = extract_thoughts_and_acts(human_conversation_hist[-1][1], search_txt=" Reason_human:")
            if not human_thoughts: human_thoughts, human_acts = extract_thoughts_and_acts(human_conversation_hist[-1][1], search_txt="")
            human_predicates_hist[human_idx].extend([f"day {day_idx} time {time_} task {k}: {human_thought}" for k, human_thought in enumerate(human_thoughts)])


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
                extracted_planning = extract_code("predicates_reflection_2", pathlib.Path(results_path) / "human/gpt_response" / f"collaboration_{collab_type}/predicates_reflection_2" / str(human_idx).zfill(5) / scene_id / day, j, collab=collab_type)
            else:
                extracted_planning = extract_code("predicates_reflection_2", pathlib.Path(results_path) / "human/llama_response" / f"collaboration_{collab_type}/predicates_reflection_2" / str(human_idx).zfill(5) / scene_id / day, j, collab=collab_type)

            # if collab_type == 1:
            #     execute_humanoid_1(env, observations, replay_dir, humanoid_rearrange_controller, human_idx, scene_id, day, time_, extracted_planning, motion_sets_list, [static_obj_room_mapping, dynamic_obj_room_mapping], [static_obj_trans_dict, dynamic_obj_trans_dict], room_dict, sentence_model, gpt=use_gpt_human)
            # elif collab_type == 2:
            #     execute_humanoid_2(env, observations, convert_helper, folder_dict, motion_dict, human_urdf_path, replay_dir, humanoid_rearrange_controller, human_idx, scene_id, day, time_, extracted_planning, motion_sets_list, npy_file_folder_list, [static_obj_room_mapping, dynamic_obj_room_mapping], [static_obj_trans_dict, dynamic_obj_trans_dict], room_dict, sentence_model, gpt=use_gpt_human)


            # =====================================================================================================================
            # Robot Inferring Intentions
            if use_gpt_human:
                video_dir_search_pattern = os.path.join(replay_dir, f"human/gpt_response/collaboration_{collab_type}/{str(human_idx).zfill(5)}/{scene_id}/{day}/*_{time_}")
            else:
                video_dir_search_pattern = os.path.join(replay_dir, f"human/llama_response/collaboration_{collab_type}/{str(human_idx).zfill(5)}/{scene_id}/{day}/*_{time_}")
            video_dir = glob.glob(video_dir_search_pattern)[0]
            
            robot_retrieved_intentions = robot_intentions_hist[human_idx][-3:] if len(robot_intentions_hist[human_idx]) >= 3 else (robot_intentions_hist[human_idx] if robot_intentions_hist[human_idx] else [])
            robot_retrieved_predicates = []

            robot_conversation_hist = intention_discovery_mllm(results_path, human_idx, scene_id, [day, j, time_], [os.path.join(video_dir, "robot_scene_camera_rgb_video"), os.path.join(video_dir, "human_third_rgb_video")], [robot_retrieved_intentions, robot_retrieved_predicates, human_thoughts[0]], [inferred_profile[human_idx], inferred_traits[human_idx]], temperature_dict, model_dict, method="main", collab=collab_type, setting=collab_setting, gpt=use_gpt_robot, start_over=start_logic_robot)
            _, pred_intention_sentence_list, robot_sampled_static_obj_dict_list = sample_obj_by_similarity(robot_conversation_hist, static_obj_room_mapping, sentence_model, top_k=30)
        
            _, data_test = create_data(pred_intention_sentence_list, [None]*intentions_num, time_, [inferred_profile[human_idx], inferred_traits[human_idx]], [robot_retrieved_intentions, robot_retrieved_predicates], data_type="intention")

            if start_logic_lora:
                if day_counter == 0:
                    answer_intentions = test_model(data_test, None, data_type="intention", pretrained=False)
                else:
                    answer_intentions = test_model(data_test, prev_lora_intention_dir, data_type="intention", pretrained=True)
                save_answers(answer_intentions, eval_dir, f"{human_idx}_{day_idx}_{j}_intention.txt")
            else:
                answer_intentions = load_answers(eval_dir, f"{human_idx}_{day_idx}_{j}_intention.txt")
            answer_intentions_within_day.extend(answer_intentions)

            # Robot gets the intention with "Yes" answer
            selected_intention_idx_list = get_intention_idx(answer_intentions)
            selected_intention_sentence_list = [pred_intention_sentence_list[idx] for idx in selected_intention_idx_list]
            selected_sampled_static_obj_dict_list = [robot_sampled_static_obj_dict_list[idx] for idx in selected_intention_idx_list]

            # Robot Inferring Predicates
            robot_thoughts, robot_acts = [], []
            data_test = []

            for k, (selected_intention_sentence, selected_sampled_static_obj_dict) in enumerate(zip(selected_intention_sentence_list, selected_sampled_static_obj_dict_list)):
                robot_retrieved_predicates = []

                robot_conversation_hist = predicates_discovery_mllm(results_path, human_idx, scene_id, [day, file_idx+k, time_], [selected_sampled_static_obj_dict, dynamic_obj_room_mapping], [robot_retrieved_intentions, robot_retrieved_predicates], selected_intention_sentence, [inferred_profile[human_idx], inferred_traits[human_idx]], robot_conversation_hist, temperature_dict, model_dict, method="main", collab=collab_type, setting=collab_setting, gpt=use_gpt_robot, start_over=start_logic_robot)
                robot_thoughts_batch, robot_acts_batch = extract_thoughts_and_acts(robot_conversation_hist[-1][1], search_txt=" Reason_human:")
                if not robot_thoughts_batch: robot_thoughts_batch, robot_acts_batch = extract_thoughts_and_acts(robot_conversation_hist[-1][1], search_txt="")
                robot_thoughts.extend(robot_thoughts_batch)
                robot_acts.extend(robot_acts_batch)

                _, data_test_batch = create_data([None, robot_thoughts_batch, extract_inhand_obj_robot(robot_acts_batch)], [None]*predicates_num, time_, [inferred_profile[human_idx], inferred_traits[human_idx]], [robot_retrieved_intentions, robot_retrieved_predicates], data_type="predicates")
                data_test.extend(data_test_batch)

            if start_logic_lora:
                if day_counter == 0:
                    answer_predicates = test_model(data_test, None, data_type="predicates", pretrained=False)
                else:
                    answer_predicates = test_model(data_test, prev_lora_predicates_dir, data_type="predicates", pretrained=True)
                save_answers(answer_predicates, eval_dir, f"{human_idx}_{day_idx}_{j}_predicate.txt")
            else:
                answer_predicates = load_answers(eval_dir, f"{human_idx}_{day_idx}_{j}_predicate.txt")
            answer_predicates_within_day.extend(answer_predicates)

            # Discussion Period: Human Judge Approving Collaborations
            intentions_approval_res = intention_approval_mllm(results_path, human_idx, scene_id, [day, j, time_], [gt_intention_sentence, pred_intention_sentence_list], temperature_dict, model_dict, method="main", collab=collab_type, setting=collab_setting, start_over=start_logic_robot)[0][1]
            intentions_approval, _ = extract_intention_approval(intentions_approval_res)

            predicates_approval = []
            for k, _ in enumerate(selected_intention_sentence_list):
                predicates_approval_res = predicate_approval_mllm(results_path, human_idx, scene_id, [day, file_idx+k, time_], human_thoughts, extract_inhand_obj_human(human_acts), robot_thoughts[k*predicates_num:(k+1)*predicates_num], extract_inhand_obj_robot(robot_acts[k*predicates_num:(k+1)*predicates_num]), temperature_dict, model_dict, method="main", collab=collab_type, setting=collab_setting, start_over=start_logic_robot)[0][1]
                predicates_approval_batch, _ = extract_predicate_approval(predicates_approval_res)
                predicates_approval.extend(predicates_approval_batch)

            category_approval = []
            for k, _ in enumerate(selected_intention_sentence_list):
                category_approval_res = category_approval_mllm(results_path, human_idx, scene_id, [day, file_idx+k, time_], human_thoughts, extract_inhand_obj_human(human_acts), robot_thoughts[k*predicates_num:(k+1)*predicates_num], extract_inhand_obj_robot(robot_acts[k*predicates_num:(k+1)*predicates_num]), temperature_dict, model_dict, method="main", collab=collab_type, setting=collab_setting, start_over=start_logic_robot)[0][1]
                category_approval_batch, _ = extract_predicate_approval(category_approval_res)
                category_approval.extend(category_approval_batch)

            # Always append the correct intentions and tasks after the discussion
            robot_intentions_hist[human_idx].append(f"{gt_intention_sentence}")
            robot_retrieved_predicates_gt = []

            for k, (robot_thought, human_thought) in enumerate(zip(robot_thoughts, human_thoughts)):
                robot_predicates_hist[human_idx].append(f"day {day_idx} time {time_} task {k}: {human_thought}")
 
            # Finetune with LoRA
            # Create intention training data
            labels = []
            for k in range(intentions_num):
                label = "Yes" if intentions_approval[k] == "yes" else "No"
                labels.append(label)
            labels_intentions_llm_within_day.extend(labels)

            data_train_intentions_batch_1, _ = create_data(pred_intention_sentence_list, labels, time_, [inferred_profile[human_idx], inferred_traits[human_idx]], [robot_retrieved_intentions, robot_retrieved_predicates_gt], data_type="intention")
            data_train_intentions_batch_2, _ = create_data([gt_intention_sentence], ["Yes"], time_, [inferred_profile[human_idx], inferred_traits[human_idx]], [robot_retrieved_intentions, robot_retrieved_predicates_gt], data_type="intention")
            data_train_intentions_batch_1.extend(data_train_intentions_batch_2)
            data_train_intentions.extend(data_train_intentions_batch_1)

            # Create predicates training data
            labels = []
            train_thoughts, train_acts= [], []

            for k in range(predicates_num * len(selected_intention_sentence_list)):
                label = "Yes" if predicates_approval[k] == "yes" else "No"
                labels_predicates_llm_within_day.append(label)
                labels_predicates_category_within_day.append("Yes" if category_approval[k] == "yes" else "No")
                train_thoughts.append(robot_thoughts[k])
                train_acts.append(robot_acts[k])
                labels.append(label)
            train_acts = extract_inhand_obj_robot(train_acts)

            # Add human predicates training data harms the performance
            # train_acts_batch = []
            # for k in range(predicates_num):
            #     labels.append("Yes")
            #     train_thoughts.append(human_thoughts[k])
            #     train_acts_batch.append(human_acts[k])
            # train_acts.extend(extract_inhand_obj_human(train_acts_batch))
            
            data_train_predicate_batch, _ = create_data([None, train_thoughts, train_acts], labels, time_, [inferred_profile[human_idx], inferred_traits[human_idx]], [robot_retrieved_intentions, robot_retrieved_predicates_gt], data_type="predicates")
            data_train_predicates.extend(data_train_predicate_batch)

            lora_intention_dir = pathlib.Path(results_path) / "lora_models" / f"collaboration_{collab_type}" / f"setting_{collab_setting}" / "main" / "intention" / str(human_idx).zfill(5) / scene_id / day
            lora_predicates_dir = pathlib.Path(results_path) / "lora_models" / f"collaboration_{collab_type}" / f"setting_{collab_setting}" / "main" / "predicates" / str(human_idx).zfill(5) / scene_id / day
            
            # Compute Evaluation Matrics
            eval_intentions_llm_within_day = calculate_accuracy_and_f1(answer_intentions_within_day, labels_intentions_llm_within_day, 2)
            eval_predicates_llm_within_day = calculate_accuracy_and_f1(answer_predicates_within_day, labels_predicates_llm_within_day, 2)
            eval_intentions_llm_across_days.append(eval_intentions_llm_within_day)
            eval_predicates_llm_across_days.append(eval_predicates_llm_within_day)

            pred_objs, gt_objs = [], []
            for k in range(predicates_num * len(selected_intention_sentence_list)):
                if answer_predicates[k].lower() == "yes":
                    pred_objs.append(robot_acts[k])
            pred_objs = extract_inhand_obj_robot(pred_objs)

            for k in range(predicates_num):
                gt_objs.append(human_acts[k])
            gt_objs = extract_inhand_obj_human(gt_objs)

            semantic_sim.append(calculate_semantic_similarity(gt_objs, pred_objs))
            avg_semantic_sim = sum(semantic_sim) / len(semantic_sim) if semantic_sim else 0
            eval_predicates_semantic_within_day = calculate_accuracy_and_f1(answer_predicates_within_day, labels_predicates_category_within_day, 2)
            eval_predicates_semantic_within_day = (*eval_predicates_semantic_within_day, avg_semantic_sim)
            eval_predicates_semantic_across_days.append(eval_predicates_semantic_within_day)
            
            # Data Visualization
            for k in range(len(predicates_approval)):
                append_evaluation_row(
                    eval_csv_data, k, predicates_num, time_,
                    gt_intention_sentence, pred_intention_sentence_list, answer_intentions,
                    human_thoughts, human_acts, robot_thoughts, robot_acts, answer_predicates,
                    profile_string, big_five, inferred_profile[human_idx], inferred_traits[human_idx],
                    (eval_big_five_across_days_latest, eval_big_five_across_days_voting),
                    intentions_approval, eval_intentions_llm_within_day,
                    predicates_approval, eval_predicates_llm_within_day,
                    category_approval, eval_predicates_semantic_within_day, method="main"
                )
            eval_csv_data.append([""] * 18)

            # keep track of the file index of the MLLM response at each time of the day
            file_idx += len(selected_intention_sentence_list)


        # Robot Inferring Human Traits
        fuzzy_traits = traits_inference_mllm(results_path, human_idx, scene_id, [day, 0, None], [robot_intentions_hist[human_idx][-13:], robot_predicates_hist[human_idx]], [inferred_profile[human_idx], inferred_traits[human_idx]], temperature_dict, model_dict, method="main", collab=collab_type, setting=collab_setting, gpt=use_gpt_robot, start_over=True)[0][1]
        inferred_traits[human_idx], inferred_profile[human_idx] = extract_scores_and_profile(fuzzy_traits)
        inferred_traits_hist[human_idx].append(inferred_traits[human_idx])
        inferred_profile_hist[human_idx].append(inferred_profile[human_idx])
        _, eval_big_five_across_days_latest = calculate_ocean_pearson_correlation(big_five, inferred_traits_hist[human_idx], latest=True)
        _, eval_big_five_across_days_voting = calculate_ocean_pearson_correlation(big_five, inferred_traits_hist[human_idx], latest=False)
        
        # Update the inferred human profile in the training data
        data_train_intentions, data_train_predicates = update_data_with_traits(data_train_intentions, data_train_predicates, [inferred_profile[human_idx], inferred_traits[human_idx]])

        # Data Visualization
        save_evaluation_results(eval_csv_path, eval_txt_path, eval_csv_data, data_train_intentions, data_train_predicates, day, method="main")
        save_results(eval_json_path, eval_intentions_llm_across_days, eval_predicates_llm_across_days, eval_predicates_semantic_across_days)
        
        # End the current environment of the scene instance
        day_counter += 1
        env.close()

        # Finetune LLM with LoRA, except after the last day
        if day_counter < len(config['days']):
            train_model(1, data_train_intentions, data_train_intentions, [day, time_], str(lora_intention_dir), data_type="intention", checkpoint_dir=None, pretrained=False)
            train_model(1, data_train_predicates, data_train_predicates, [day, time_], str(lora_predicates_dir), data_type="predicates", checkpoint_dir=None, pretrained=False)
    