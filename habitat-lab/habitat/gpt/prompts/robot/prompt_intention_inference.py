import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query
import cv2


def infer_intention_response(time_, intention):
    contents = f"Time: {time_}\nIntention: {intention}"
    return contents


def infer_intention_prompt(time_, motion_description, fuzzy_traits, retrieved_memory, collab=2):
    if collab == 1:
        contents = f"""
        Input:
        1.	Human task: {motion_description}.
        2.  Current time: {time_}.
        3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
        4.  Inferred human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).
        5.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

        You are a robot assisting a human. Identify the human's intention.

        Instructions:
        1.  Map the human task to a higher-level intention without mentioning the specific motion.
        2.	Intention must align with human Big 5 scores and reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects.
        3.	Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
        4.  Intention must have temporal dependence but be non-repetitive with the intentions at previous times in the input.

        Write in the following format. Do not output anything else:
        Time: xxx am/pm
        Intention: basic descriptions.
        """
    
    elif collab == 2:
        contents = f"""
        Input:
        1.	Descriptions of sequence of images showing human motion from your and human's perspectives: {motion_description}.
        2.  Current time: {time_}.
        3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
        4.  Inferred human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).
        5.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

        You are a robot assisting a human. Identify the human's intention.

        Instructions:
        1.  Map the observed human motion to a higher-level intention without mentioning the specific motion.
        2.	Intention must align with human Big 5 scores and reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects.
        3.	Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
        4.  Intention must have temporal dependence but be non-repetitive with the intentions at previous times in the input.

        Write in the following format. Do not output anything else:
        Time: xxx am/pm
        Intention: basic descriptions.
        """
    return contents


def infer_intention(time_, retrieved_memory, fuzzy_traits, motion_description, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, collab=2):

    intention_user_contents_filled = infer_intention_prompt(time_, motion_description, fuzzy_traits, retrieved_memory, collab=collab)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/intention_inference.json"

        print("=" * 50)
        print("=" * 20, "Inferring Intention", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [(intention_user_contents_filled, [])], [], save_path, model_dict['finetuning'], temperature_dict['finetuning'], debug=False)
        print()
        print("DEBUG", model_dict['finetuning'])
        print()

    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        intention_response = json_data["res"]
        print(intention_response)
        print()
        
    return intention_user_contents_filled, json_data["res"] 
