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

from transformers import (AutoTokenizer,
                          MistralForSequenceClassification, 
                          BitsAndBytesConfig, 
                          Trainer, 
                          TrainingArguments)
from datasets import Dataset, DatasetDict, load_dataset
from peft import (LoraConfig, 
                  PeftConfig, 
                  PeftModel, 
                  get_peft_model,
                  prepare_model_for_kbit_training)
from sklearn.metrics import f1_score
from accelerate import dispatch_model




def set_seed(seed: int):
    """
    Set the seed for reproducibility across various libraries.

    Parameters:
    seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If you are using a GPU, you should also set the seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
        
        # Ensure deterministic behavior in CUDA operations (optional)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def temperature_scaling(scores, temperature=0.5):
    scores = np.array(scores)
    scaled_scores = np.exp(np.log(scores + 1e-9) / temperature)
    scaled_scores = scaled_scores / scaled_scores.max()
    return scaled_scores.tolist()


def temperature_scaling_single_score(score_list, temperature=2.0, epsilon=1e-9):
    score = np.clip(score_list[0], epsilon, 1 - epsilon)  # Clip score to avoid 0 or 1
    logit = np.log(score / (1 - score))  # Convert the probability to logit
    scaled_logit = logit / temperature  # Apply temperature scaling
    scaled_score = 1 / (1 + np.exp(-scaled_logit))  # Convert back to probability
    return [scaled_score]  # Return as a list with a single item


def add_constant_scaling(scores, C=0.4):
    adjusted_scores = [max(min(score + C, 1), 0) for score in scores]
    return adjusted_scores


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_result = f1_score(labels, preds, average='weighted')
    return {'f1_score': f1_result}


def select_model(checkpoint_dir=None, pretrained=True):
    model_checkpoint = checkpoint_dir if pretrained else 'mistralai/Mistral-7B-v0.1'
        
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit = True,
    #     bnb_4bit_quant_type = 'nf4',
    #     bnb_4bit_compute_dtype = torch.bfloat16,
    #     bnb_4bit_use_double_quant = True,
    # )


    model = MistralForSequenceClassification.from_pretrained(
        model_checkpoint,
        use_auth_token=os.environ['HUGGINGFACE_TOKEN'],  
        num_labels=2,
        # quantization_config=bnb_config,
        device_map='balanced',
        low_cpu_mem_usage=True,  # Ensure this is set to True for 4-bit/8-bit models
        trust_remote_code=True
    )
    
    # Lora Configuration
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=2,
        bias='none',
        task_type='SEQ_CLS',
        target_modules=['q_proj', 'v_proj']
    )
    model = get_peft_model(model, peft_config)
    
    return model


def create_tokenizer(model_checkpoint):
    # Load & Tokenize Data
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token
    
    return tokenizer


def train_model(epoch, data_train, data_test, output_path, checkpoint_dir=None, pretrained=True):
    model = select_model(checkpoint_dir=checkpoint_dir, pretrained=pretrained)
    model.train()
    output_dir = str(pathlib.Path(output_path).parent)
    tokenizer = create_tokenizer(checkpoint_dir) if pretrained else create_tokenizer('mistralai/Mistral-7B-v0.1')

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True)

    train_ds = Dataset.from_dict({"text": [d["text"] for d in data_train], "label": [d["label"] for d in data_train]})
    test_ds = Dataset.from_dict({"text": [d["text"] for d in data_test], "label": [d["label"] for d in data_test]})

    ds_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    ds = ds_dict["train"]
    ds = ds.map(tokenize, batched=True)

    # Fine-tune LLM 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        num_train_epochs=epoch,
        weight_decay=0.01,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        checkpoint_path = checkpoints[0] 

        if os.path.exists(output_path):
            # Remove all contents of output_path, but not the directory itself
            for item in os.listdir(output_path):
                item_path = os.path.join(output_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        os.rename(os.path.join(output_dir, checkpoint_path), output_path)



def test_model(data_test, checkpoint_dir, cls_type="traits", pretrained=True):
    model = select_model(checkpoint_dir=checkpoint_dir, pretrained=pretrained)
    model.eval()
    tokenizer = create_tokenizer(checkpoint_dir) if pretrained else create_tokenizer('mistralai/Mistral-7B-v0.1')

    confidences = []
    for data in data_test:
        text = data["text"]
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence_scores = probs.numpy().tolist()[0]
            confidences.append(confidence_scores[1])  # label 1 is the acceptance

    if len(confidences) > 1:
        if cls_type == "traits":
            confidences = add_constant_scaling(temperature_scaling(confidences, temperature=0.5), C=-0.4)
        else:
            confidences = add_constant_scaling(temperature_scaling(confidences, temperature=1.0), C=0.4)
    else:
        confidences = temperature_scaling_single_score(confidences, temperature=10.0)

    print()
    print(checkpoint_dir)
    print(54321, f"Confidence Scores: {confidences}")
    print()

    return confidences


def create_data(texts, labels, time_, traits, hist, data_type="intention", cls_type="traits"):
    if cls_type == "traits":
        if data_type == "intention":
            if labels is None: labels = 0
            data_train = [{"text": f"Big Five human traits: {traits}. Time: {time_}. Intention: {texts}", "label": {labels}}]
        elif data_type == "predicates":
            data_train = []
            thoughts, acts = texts[0], texts[1]
            for (thought, act, label) in zip(thoughts, acts, labels):
                if label is None: label = 0
                data_train.append({"text": f"Big Five human traits: {traits}. Time: {time_}. Task: {thought} Needed object: {act}.", "label": {label}})
    
    elif cls_type == "temporal":
        intentions_hist, predicates_hist = hist[0], hist[1]
        if data_type == "intention":
            if labels is None: labels = 0
            data_train = [{"text": f"Most relevant intentions at previous times: {intentions_hist}. Most relevant tasks at previous times: {predicates_hist}. Current time: {time_}. Current intention: {texts}", "label": {labels}}]
        elif data_type == "predicates":
            data_train = []
            thoughts, acts = texts[0], texts[1]
            for (thought, act, label) in zip(thoughts, acts, labels):
                if label is None: label = 0
                data_train.append({"text": f"Most relevant intentions at previous times: {intentions_hist}. Most relevant tasks at previous times: {predicates_hist}. Current time: {time_}. Current task: {thought} Needed object: {act}.", "label": {label}})
    
    data_test = data_train
    # print()
    # print(12345, data_test)
    # print()
    return data_train, data_test
    
