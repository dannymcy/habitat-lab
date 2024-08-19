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
from habitat.gpt.prompts.utils import load_response

from sentence_transformers import SentenceTransformer




def calculate_ocean_mse(ocean1, ocean2):
    """
    Calculate the Mean Squared Error (MSE) between two OCEAN matrices.

    :param ocean1: Dictionary with OCEAN traits as keys and their corresponding scores as values.
    :param ocean2: Dictionary with OCEAN traits as keys and their corresponding scores as values.
    :return: The Mean Squared Error (MSE) between the two OCEAN matrices.
    """
    mse = 0.0
    for trait in ocean1:
        mse += (ocean1[trait] - ocean2[trait]) ** 2
    
    mse /= len(ocean1)
    
    return mse


def traits_inference_gpt4(data_path, human_id, scene_id, time_tuple, retrieved_memory, fuzzy_traits, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "judge/traits_inference" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []
    file_idx, time_ = time_tuple

    if start_over:
        user, res = infer_traits(retrieved_memory, fuzzy_traits, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        time.sleep(20)
    else:
        user, res = infer_traits(retrieved_memory, fuzzy_traits, output_dir, existing_response=load_response("traits_inference", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
    conversation_hist.append([user, res])

    return conversation_hist

