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

from habitat.gpt.prompts.robot.prompt_intention_discovery import discover_intention
from habitat.gpt.prompts.robot.prompt_predicates_discovery import discover_predicates
from habitat.gpt.prompts.utils import load_response

from sentence_transformers import SentenceTransformer




def extract_frames(video_path, output_dir):
    ####################
    # SlowFast transform
    ####################

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    side_size = 512
    crop_size = 512
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    num_frames = 4
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    sampling_rate = 2
    alpha = 2
    # The duration of the input clip is also specific to the model.
    clip_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / frames_per_second

    cap.release()

    class PackPathway(torch.nn.Module):
        """
        Transform for converting video frames as a list of tensors. 
        """
        def __init__(self):
            super().__init__()
            
        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // alpha
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
            return frame_list

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                # Lambda(lambda x: x/255.0),
                # NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    ) 

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration 

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    os.makedirs(output_dir, exist_ok=True)
    frames_data = video_data["video"][1].to(torch.uint8)  # [0] slow, 1 [fast]

    for i in range(frames_data.shape[1]):
        if i == 0: continue
        frame_data = frames_data[:, i, :, :]  # Shape: [3, 512, 512]
        frame_image = to_pil_image(frame_data)  # Convert tensor to PIL image
        output_path = os.path.join(output_dir, f"{i}.png")  # Save with index-based name
        frame_image.save(output_path)
        print(f"Frame {i} saved to {output_path}")


def fluctuate_vector(vector, mean=0.0, std=0.5):
    fluct_x = np.random.normal(mean, std)
    fluct_y = np.random.normal(mean, std)
    fluct_z = np.random.normal(mean, std)

    vector.x += fluct_x
    vector.y += fluct_y
    vector.z += fluct_z

    return vector


def find_closest_objects(object_trans, obj_trans_dict_to_search, k=5):
    def euclidean_distance(vec1, vec2):
        return np.linalg.norm(np.array(vec1) + np.array(vec2))

    distances = []
    for name, (obj_id, trans) in obj_trans_dict_to_search.items():
        dist = euclidean_distance(object_trans, trans)
        distances.append((dist, trans))

    # Sort based on distance
    distances.sort(key=lambda x: x[0], reverse=True)  # reverse returns the furthest objects

    # Get the top k closest objects
    search_trans = [trans for _, trans in distances[:k]]

    return search_trans

    
def intention_discovery_gpt4(data_path, scene_id, time_, video_dir, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "robot/intention_discovery" / scene_id
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = discover_intention(time_, video_dir, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        time.sleep(20)
    else:
        user, res = discover_intention(time_, video_dir, output_dir, existing_response=load_response("intention_discovery", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
    conversation_hist.append([user, res])

    return conversation_hist
    

def predicates_discovery_gpt4(data_path, scene_id, time_, conversation_hist, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "robot/predicates_discovery" / scene_id
    os.makedirs(output_dir, exist_ok=True)

    if start_over:
        user, res = discover_predicates(time_, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        time.sleep(20)
    else:
        user, res = discover_predicates(time_, output_dir, existing_response=load_response("predicates_discovery", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
    conversation_hist.append([user, res])

    return conversation_hist


def get_feedback(robot_discovery, human_gt):
    return