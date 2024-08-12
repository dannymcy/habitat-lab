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

from habitat.gpt.prompts.human.prompt_traits_summary import summarize_traits
from habitat.gpt.prompts.human.prompt_intention_proposal import propose_intention
from habitat.gpt.prompts.human.prompt_predicates_proposal import propose_predicates
from habitat.gpt.prompts.human.prompt_predicates_reflection_1 import reflect_predicates_1
from habitat.gpt.prompts.human.prompt_predicates_reflection_2 import reflect_predicates_2
from habitat.gpt.prompts.human.prompt_collaboration_proposal import propose_collaboration
from habitat.gpt.prompts.utils import load_response, extract_times, extract_intentions, extract_code

from sentence_transformers import SentenceTransformer




def calculate_recency_scores(times, selected_time, predicates_num, decay_factor=0.95):
    # Find the index of the selected time
    selected_index = times.index(selected_time)
    
    # Calculate the decaying scores
    intentions_recency = []
    for i in range(selected_index - 1, -1, -1):
        recency = decay_factor ** (selected_index - 1 - i)
        intentions_recency.append(recency)
    
    intentions_recency.reverse()
    predicates_recency = [recency for recency in intentions_recency for _ in range(predicates_num)]
    return intentions_recency, predicates_recency


def calculate_relevance_scores(gt_text, activity_list):
    """
    Compute similarity scores between a given text and a list of activities.
    If activity_list is None, return None.
    """
    if not activity_list:
        return []

    def compute_similarity(gt_text, activity_list):
        """
        Compute semantic similarity between the given text and activities.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Encode the ground truth text and activity list
        gt_embedding = model.encode([gt_text])  # Note the input is a list
        activity_embeddings = model.encode(activity_list)
        
        # Compute similarities
        similarities = np.dot(activity_embeddings, gt_embedding.T).flatten()  # Using dot product for cosine similarity
        return similarities

    # Compute similarity scores
    similarities = compute_similarity(gt_text, activity_list)
    
    return similarities


def retrieve_memory(gt_text, activity_list, times, selected_time, predicates_num, decay_factor=0.95, top_k=5, retrieve_type="intention"):
    # Check if activity_list is None
    if not activity_list:
        return []

    intentions_recency, predicates_recency = calculate_recency_scores(times, selected_time, predicates_num, decay_factor=0.95)
    relevance_scores = calculate_relevance_scores(gt_text, activity_list)
    recency_scores = intentions_recency if retrieve_type == "intention" else predicates_recency

    if len(relevance_scores) != len(recency_scores):
        raise ValueError("The lengths of relevance scores and recency scores must match.")
    
    combined_scores = [relevance * recency for relevance, recency in zip(relevance_scores, recency_scores)]

    # Pair activities with their combined scores
    activity_scores = list(zip(activity_list, combined_scores))

    # Sort activities by combined score in descending order
    activity_scores.sort(key=lambda x: x[1], reverse=True)

    # Select the top_k activities, or all if there are fewer than top_k
    top_activities = activity_scores[:top_k]

    # Extract just the activities from the top_k list
    top_activity_list = [activity for activity, score in top_activities]
    
    return top_activity_list


def traits_summary_gpt4(data_path, human_id, scene_id, profile_string, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "human/traits_summary" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = summarize_traits(profile_string, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        time.sleep(20)
    else:
        user, res = summarize_traits(profile_string, output_dir, existing_response=load_response("traits_summary", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
    conversation_hist.append([user, res])

    return conversation_hist


def intention_proposal_gpt4(data_path, human_id, scene_id, time_tuple, retrieved_memory, room_list, profile_string, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "human/intention_proposal" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    file_idx, time_ = time_tuple
    conversation_hist = []

    if start_over:
        user, res = propose_intention(time_, room_list, profile_string, retrieved_memory, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        time.sleep(20)
    else:
        user, res = propose_intention(time_, room_list, profile_string, retrieved_memory, output_dir, existing_response=load_response("intention_proposal", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
    conversation_hist.append([user, res])

    return conversation_hist


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


def create_motion_sets(npy_file_folder_list, human_urdf_path, sample_motion=None, update=False):
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
        if sample_motion is None:
            # Get all filenames ending with .npy in the folder
            filenames = [filename for filename in os.listdir(folder) if filename.endswith(".npy")]
        else:
            # Concatenate the folder path with the name of each motion in sample_motion, with the extension .npy
            filenames = [f"{sample_motion}.npy"]
        
        for filename in filenames:
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


def sample_obj_by_similarity(conversation_hist, object_dict, top_k=30):
    """
    Extract intentions from conversation history and compute similarity scores with object names.
    """
    def compute_similarity(intention_sentences, object_dict):
        """
        Compute semantic similarity between intention sentences and object names.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Get object names with rooms
        object_names = [f"{name} in {data[1]}" for name, data in object_dict.items()]
        
        # Encode sentences
        intention_embeddings = model.encode(intention_sentences)
        object_embeddings = model.encode(object_names)
        
        # Compute similarities
        similarities = model.similarity(intention_embeddings, object_embeddings)
        return similarities

    # Extract times and intentions
    times = extract_times(conversation_hist[0][1])
    intention_sentences = extract_intentions(conversation_hist[0][1])
    
    # Compute similarity
    similarities = compute_similarity(intention_sentences, object_dict)
    # print()
    # print(similarities)
    # print()
    
    # Sample top K objects for each intention
    sampled_objects_dict_list = []
    object_names = list(object_dict.keys())

    for sim in similarities:
        top_indices = np.argsort(-sim)[:top_k]  # Get indices of top K similarities
        sampled_objects = {object_names[idx]: object_dict[object_names[idx]] for idx in top_indices}
        sampled_objects_dict_list.append(sampled_objects)
        # print()
        # print(sampled_objects)

    return times, intention_sentences, sampled_objects_dict_list


def sample_motion_by_similarity(conversation_hist, motion_list, top_k=5):
    """
    Extract intentions from conversation history and compute similarity scores with motions.
    """
    def compute_similarity(intention_sentences, motion_list):
        """
        Compute semantic similarity between intention sentences and motions.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Encode sentences
        intention_embeddings = model.encode(intention_sentences)
        motion_embeddings = model.encode(motion_list)
        
        # Compute similarities
        similarities = model.similarity(intention_embeddings, motion_embeddings)
        return similarities

    # Extract intentions
    intention_sentences = extract_intentions(conversation_hist[0][1])
    
    # Compute similarity
    similarities = compute_similarity(intention_sentences, motion_list)
    
    # Sample top K motions for each intention
    sampled_motion_list = []

    for sim in similarities:
        top_indices = np.argsort(-sim)[:top_k]  # Get indices of top K similarities
        sampled_motion = [motion_list[idx] for idx in top_indices]
        sampled_motion_list.append(sampled_motion)

    return intention_sentences, sampled_motion_list


def predicates_proposal_gpt4(data_path, human_id, scene_id, time_tuple, retrieved_memory, sampled_motion_list, obj_room_mapping, profile_string, conversation_hist, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "human/predicates_proposal" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    file_idx, time_ = time_tuple
    
    if start_over:
        user, res = propose_predicates(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        time.sleep(20)
    else:
        user, res = propose_predicates(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_dir, existing_response=load_response("predicates_proposal", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
    conversation_hist.append([user, res])

    return conversation_hist


def predicates_reflection_1_gpt4(data_path, human_id, scene_id, time_tuple, retrieved_memory, sampled_motion_list, obj_room_mapping, profile_string, conversation_hist, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "human/predicates_reflection_1" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    file_idx, time_ = time_tuple

    if start_over:
        user, res = reflect_predicates_1(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        time.sleep(20)
    else:
        user, res = reflect_predicates_1(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_dir, existing_response=load_response("predicates_reflection", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
    conversation_hist.append([user, res])

    return conversation_hist


def predicates_reflection_2_gpt4(data_path, human_id, scene_id, time_tuple, retrieved_memory, sampled_motion_list, obj_room_mapping, profile_string, conversation_hist, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "human/predicates_reflection_2" / str(human_id).zfill(5) / scene_id
    os.makedirs(output_dir, exist_ok=True)
    file_idx, time_ = time_tuple

    if start_over:
        user, res = reflect_predicates_2(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        time.sleep(20)
    else:
        user, res = reflect_predicates_2(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_dir, existing_response=load_response("predicates_reflection", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
    conversation_hist.append([user, res])
    
    return conversation_hist


def most_similar_motion(free_motion, motion_list, top_k=1):
    def compute_similarity(free_motion, motion_list):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Encode sentences
        emb = model.encode(free_motion)
        motion_embeddings = model.encode(motion_list)
        
        # Compute similarities
        similarities = model.similarity(emb, motion_embeddings)
        return similarities
    
    # Compute similarity
    similarities = compute_similarity(free_motion, motion_list)
    
    # Sample top K motions for each intention
    sampled_motion_list = []

    for sim in similarities:
        top_indices = np.argsort(-sim)[:top_k]  # Get indices of top K similarities
        sampled_motion = [motion_list[idx] for idx in top_indices]
        sampled_motion_list.append(sampled_motion)

    return sampled_motion


def collaboration_proposal_gpt4(data_path, scene_id, time_, sampled_motion_list, extracted_planning, predicate, thought, act, conversation_hist, temperature_dict, model_dict, start_over=False):
    output_dir = pathlib.Path(data_path) / "gpt4_response" / "human/collaboration_proposal" / scene_id
    os.makedirs(output_dir, exist_ok=True)

    if start_over:
        user, res = propose_collaboration(time_, sampled_motion_list, extracted_planning, predicate, thought, act, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        time.sleep(20)
    else:
        user, res = propose_collaboration(time_, sampled_motion_list, extracted_planning, predicate, thought, act, output_dir, existing_response=load_response("collaboration_proposal", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
    conversation_hist.append([user, res])

    return conversation_hist