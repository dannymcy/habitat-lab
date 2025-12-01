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

from habitat.gpt.prompts.robot.prompt_intention_inference import infer_intention
from habitat.gpt.prompts.robot.prompt_predicates_discovery import discover_predicates
from habitat.gpt.prompts.robot.prompt_traits_inference import infer_traits
from habitat.gpt.prompts.robot.prompt_motion_description import describe_motion
from habitat.gpt.prompts.robot.prompt_intention_discovery import discover_intention
from habitat.gpt.prompts.utils import load_response

from sentence_transformers import SentenceTransformer



def intention_discovery_mllm(data_path, human_id, scene_id, time_tuple, video_dirs, retrieved_memory, fuzzy_traits, temperature_dict, model_dict, method="main", collab=2, setting=1, gpt=True, start_over=False):
    day, file_idx, time_ = time_tuple
    if gpt:
        output_dir = pathlib.Path(data_path) / "robot/gpt_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/intention_discovery" / str(human_id).zfill(5) / scene_id / day
    else:
        output_dir = pathlib.Path(data_path) / "robot/llama_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/intention_discovery" / str(human_id).zfill(5) / scene_id / day

    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = discover_intention(time_, retrieved_memory, fuzzy_traits, video_dirs, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None, method=method, collab=collab, gpt=gpt)
        if gpt: time.sleep(20)
    else:
        user, res = discover_intention(time_, retrieved_memory, fuzzy_traits, video_dirs, output_dir, existing_response=load_response("intention_discovery", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None, method=method, collab=collab, gpt=gpt)
    conversation_hist.append([user, res])

    return conversation_hist


def predicates_discovery_mllm(data_path, human_id, scene_id, time_tuple, obj_room_mapping, retrieved_memory, selected_intention_sentence, fuzzy_traits, conversation_hist, temperature_dict, model_dict, method="main", collab=2, setting=1, gpt=True, start_over=False):
    day, file_idx, time_ = time_tuple
    if gpt:
        output_dir = pathlib.Path(data_path) / "robot/gpt_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/predicates_discovery" / str(human_id).zfill(5) / scene_id / day
    else:
        output_dir = pathlib.Path(data_path) / "robot/llama_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/predicates_discovery" / str(human_id).zfill(5) / scene_id / day
    os.makedirs(output_dir, exist_ok=True)

    if start_over:
        user, res = discover_predicates(time_, retrieved_memory, selected_intention_sentence, fuzzy_traits, obj_room_mapping, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist, method=method, collab=collab, gpt=gpt)
        if gpt: time.sleep(20)
    else:
        user, res = discover_predicates(time_, retrieved_memory, selected_intention_sentence, fuzzy_traits, obj_room_mapping, output_dir, existing_response=load_response("predicates_discovery", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist, method=method, collab=collab, gpt=gpt)
    conversation_hist.append([user, res])

    return conversation_hist


def traits_inference_mllm(data_path, human_id, scene_id, time_tuple, retrieved_memory, fuzzy_traits, temperature_dict, model_dict, method="main", collab=2, setting=1, gpt=True, start_over=False):
    day, file_idx, time_ = time_tuple
    if gpt:
        output_dir = pathlib.Path(data_path) / "robot/gpt_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/traits_inference" / str(human_id).zfill(5) / scene_id / day
    else:
        output_dir = pathlib.Path(data_path) / "robot/llama_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/traits_inference" / str(human_id).zfill(5) / scene_id / day
    
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = infer_traits(time_, retrieved_memory, fuzzy_traits, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None, gpt=gpt)
        if gpt: time.sleep(20)
    else:
        user, res = infer_traits(time_, retrieved_memory, fuzzy_traits, output_dir, existing_response=load_response("traits_inference", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None, gpt=gpt)
    conversation_hist.append([user, res])

    return conversation_hist


def motion_description_mllm(data_path, human_id, scene_id, time_tuple, video_dirs, temperature_dict, model_dict, method="finetuning", collab=2, setting=1, start_over=False):
    day, file_idx, time_ = time_tuple
    output_dir = pathlib.Path(data_path) / "robot/gpt_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/motion_description" / str(human_id).zfill(5) / scene_id / day
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = describe_motion(time_, video_dirs, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        time.sleep(20)
    else:
        user, res = describe_motion(time_, video_dirs, output_dir, existing_response=load_response("motion_description", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
    conversation_hist.append([user, res])

    return conversation_hist


def intention_inference_mllm(data_path, human_id, scene_id, time_tuple, motion_description, retrieved_memory, fuzzy_traits, temperature_dict, model_dict, method="finetuning", collab=2, setting=1, start_over=False):
    day, file_idx, time_ = time_tuple
    output_dir = pathlib.Path(data_path) / "robot/gpt_response" / f"collaboration_{collab}/setting_{setting}" / f"{method}/intention_inference" / str(human_id).zfill(5) / scene_id / day
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = infer_intention(time_, retrieved_memory, fuzzy_traits, motion_description, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None, collab=collab)
        time.sleep(20)
    else:
        user, res = infer_intention(time_, retrieved_memory, fuzzy_traits, motion_description, output_dir, existing_response=load_response("intention_inference", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None, collab=collab)
    conversation_hist.append([user, res])

    return conversation_hist
    

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

    transform = ApplyTransformToKey(
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
