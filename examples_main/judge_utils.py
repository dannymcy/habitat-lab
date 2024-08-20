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
from collections import Counter

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

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)

from habitat.gpt.prompts.judge.prompt_traits_inference import infer_traits
from habitat.gpt.prompts.judge.prompt_collaboration_approval import approve_collaboration
from habitat.gpt.prompts.utils import load_response

from sentence_transformers import SentenceTransformer




def calculate_ocean_mse(ocean1, ocean2_list):
    """
    Calculate the Mean Squared Error (MSE) between a ground truth OCEAN matrix and a list of OCEAN matrices.

    :param ocean1: Dictionary with OCEAN traits as keys and their corresponding scores as values (ground truth).
    :param ocean2_list: List of dictionaries where each dictionary has OCEAN traits as keys and their corresponding scores as values.
    :return: The Mean Squared Error (MSE) between the ground truth OCEAN matrix and the majority-voted OCEAN matrix.
    """
    def round_to_nearest_half(value):
        """
        Round a value to the nearest 0.5 increment.
        """
        return round(value * 2) / 2
        
    # Initialize the majority-voted OCEAN dictionary
    majority_voted_ocean = {}

    # If there is only one dictionary in the ocean2_list, take that as the majority voted result
    if len(ocean2_list) == 1:
        ocean2_list = [{k: round_to_nearest_half(v) for k, v in ocean2_list[0].items()}]

    # Iterate over each trait in the OCEAN model
    for trait in ocean1:
        # Get all the rounded values for this trait from each dictionary in ocean2_list
        rounded_values = [round_to_nearest_half(ocean[trait]) for ocean in ocean2_list]
        
        # Take the majority vote for this trait
        majority_vote = Counter(rounded_values).most_common(1)[0][0]
        majority_voted_ocean[trait] = majority_vote

    # Calculate MSE between ocean1 and the majority-voted ocean
    mse = 0.0
    for trait in ocean1:
        mse += (ocean1[trait] - majority_voted_ocean[trait]) ** 2
    
    mse /= len(ocean1)
    
    return majority_voted_ocean, mse


def calculate_confidence_avg(confidence_hist):
    if not confidence_hist:
        return 0.0

    return sum(confidence_hist) / len(confidence_hist)


def calculate_accuracy(results):
    if not results:  # Check if the list is empty
        return 0.0

    correct_count = results.count('yes')
    total_count = len(results)
    
    accuracy = correct_count / total_count
    return accuracy


def traits_inference_gpt4(data_path, human_id, scene_id, time_tuple, retrieved_memory, fuzzy_traits, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "judge/traits_inference" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []
    file_idx, time_ = time_tuple

    if start_over:
        user, res = infer_traits(time_, retrieved_memory, fuzzy_traits, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        time.sleep(20)
    else:
        user, res = infer_traits(time_, retrieved_memory, fuzzy_traits, output_dir, existing_response=load_response("traits_inference", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
    conversation_hist.append([user, res])

    return conversation_hist


def collaboration_approval_gpt4(data_path, human_id, scene_id, time_tuple, intentions, human_thoughts, human_acts, robot_thoughts, robot_acts, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "judge/collaboration_approval" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []
    file_idx, time_ = time_tuple

    if start_over:
        user, res = approve_collaboration(time_, intentions, human_thoughts, human_acts, robot_thoughts, robot_acts, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        time.sleep(20)
    else:
        user, res = approve_collaboration(time_, intentions, human_thoughts, human_acts, robot_thoughts, robot_acts, output_dir, existing_response=load_response("collaboration_approval", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
    conversation_hist.append([user, res])

    return conversation_hist