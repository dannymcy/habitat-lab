import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query, llm_local_inference
import cv2


def discover_intention_prompt_1(time_, first_predicate, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	First human task: {first_predicate} of the intention and image showing human motion.
    2.  Current time: {time_}.
    3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    4.  Inferred human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).
    5.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

    You are a robot assisting a human. Identify 3 possible human intentions based on the current time and visual observations.

    Instructions:
    1.  Map the human task to 3 possible high-level intentions at the current time (without mentioning the specific motion).
    2.	Intention must align with human Big 5 scores and reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects.
    3.	Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
    4.  Intention must have temporal dependence but be non-repetitive with the intentions at previous times in the input.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention 1: basic descriptions.
    Reason_human: detailed descriptions of why it follows the Big 5 scores and profile.
    Reason_intentions: detailed descriptions of why it has temporal dependence with the previous, relevant intentions. 
    Reason_txt: detailed descriptions with respect to the human task.

    Intention 2: basic descriptions.
    Reason_human: ...
    Reason_intentions: ...
    Reason_txt: ...

    Intention 3: basic descriptions.
    Reason_human: ...
    Reason_intentions: ...
    Reason_txt: ...
    """
    return contents


def discover_intention_prompt_2(time_, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	An image showing human motion from your perspective.
    2.  Current time: {time_}.
    3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    4.  Inferred human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).
    5.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

    You are a robot assisting a human. Identify 5 possible human intentions based on the current time and visual observations.

    Instructions:
    1.  Map the observed human motion to 5 possible high-level intentions at the current time (without mentioning the specific motion).
    2.	Intention must align with human Big 5 scores and reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects.
    3.	Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
    4.  Intention must have temporal dependence but be non-repetitive with the intentions at previous times in the input.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention 1: basic descriptions.
    Reason_human: detailed descriptions of why it follows the Big 5 scores and profile.
    Reason_intentions: detailed descriptions of why it has temporal dependence with the previous, relevant intentions. 
    Reason_vis: detailed descriptions with respect to the visual cues.

    Intention 2: basic descriptions.
    Reason_human: ...
    Reason_intentions: ...
    Reason_vis: ...

    Intention 3: basic descriptions.
    Reason_human: ...
    Reason_intentions: ...
    Reason_vis: ...

    Intention 4: basic descriptions.
    Reason_human: ...
    Reason_intentions: ...
    Reason_vis: ...

    Intention 5: basic descriptions.
    Reason_human: ...
    Reason_intentions: ...
    Reason_vis: ...
    """
    return contents


def discover_intention_prompting_prompt_1(time_, first_predicate, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	First human task: {first_predicate} of the intention and image showing human motion.
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
    Reason_human: detailed descriptions of why it follows the Big 5 scores and profile.
    Reason_intentions: detailed descriptions of why it has temporal dependence with the previous, relevant intentions. 
    Reason_txt: detailed descriptions with respect to the human task.
    """
    return contents


def discover_intention_prompting_prompt_2(time_, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	An image showing human motion from your perspective.
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
    Reason_human: detailed descriptions of why it follows the Big 5 scores and profile.
    Reason_intentions: detailed descriptions of why it has temporal dependence with the previous, relevant intentions. 
    Reason_vis: detailed descriptions with respect to the visual cues.
    """
    return contents


def discover_intention_ag_intent_prompt_1(time_, first_predicate, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	First human task: {first_predicate} of the intention and image showing human motion.
    2.  Current time: {time_}.
    3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    4.  Inferred human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).
    5.  A dict mapping static (fixed) furnitures to their IDs and rooms: {retrieved_memory[2]}.
    6.  A dict mapping dynamic objects to their IDs and rooms: {retrieved_memory[3]}.
    7.  Most relevant human tasks discovered at previous times: {retrieved_memory[1]} (ignore if empty—this means it's your first collaboration with this human).

    You are a robot assisting a human.

    Instructions:
    1.  Infer 2 additional tasks based on the human intention and initial task.
    2.	Task type: For each task, pick a dynamic object and place on a fixed, static object (static objects cannot be moved).
    3.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    4.  Ensure each task includes Act: [static_obj_name: your chosen static obj, dynamic_obj_name: your chosen dynamic obj] (exact format).

    Write in the following format (1 given human task + 2 inferred by you, so EXACTLY 3 tasks in the Tasks list). Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: detailed descriptions of the task. Reason_human: why it alignes with your Big 5 scores and profile. Reason_tasks: how it depends on previous, relevant tasks. Act: [static_obj_name: xxx, dynamic_obj_name: yyy]
    2. ...
    """
    return contents


def discover_intention_ag_intent_prompt_2(time_, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	An image showing human motion from your perspective.
    2.  Current time: {time_}.
    3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    4.  Inferred human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).
    5.  Most relevant human tasks discovered at previous times: {retrieved_memory[1]} (ignore if empty—this means it's your first collaboration with this human).

    You are a robot assisting a human. Identify 5 possible human tasks based on the current time and visual observations.

    Instructions:
    1.  Map the observed human motion to 5 tasks.
    2.	Task type: For each human task, provide one small, handable object from a magical box.
    3.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    4. 	All objects are rigid and cannot deform, disassemble, or transform.
    5.  Ensure each task includes Act: [obj_name: your chosen object] (exact format).

    Write in the following format. Make sure there are 5 tasks. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: detailed descriptions of the task. Reason_human: why it alignes with your Big 5 scores and profile. Reason_tasks: how it depends on previous, relevant tasks. Act: [obj_name: xxx]
    2. ...
    """
    return contents


def discover_intention(time_, retrieved_memory, fuzzy_traits, video_dirs, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, method="main", collab=2, gpt=True):

    if collab == 1:
        first_predicate = retrieved_memory[-1]
        if method == "prompting":
            intention_user_contents_filled = discover_intention_prompting_prompt_1(time_, first_predicate, fuzzy_traits, retrieved_memory)
        elif method == "ag_intent":
            intention_user_contents_filled = discover_intention_ag_intent_prompt_1(time_, first_predicate, fuzzy_traits, retrieved_memory)
        elif method in ["main", "ag_human", "random_"]:
            intention_user_contents_filled = discover_intention_prompt_1(time_, first_predicate, fuzzy_traits, retrieved_memory)
    elif collab == 2:
        if method == "prompting":
            intention_user_contents_filled = discover_intention_prompting_prompt_2(time_, fuzzy_traits, retrieved_memory)
        elif method == "ag_intent":
            intention_user_contents_filled = discover_intention_ag_intent_prompt_2(time_, fuzzy_traits, retrieved_memory)
        elif method in ["main", "ag_human", "random_"]:
            intention_user_contents_filled = discover_intention_prompt_2(time_, fuzzy_traits, retrieved_memory)

    if gpt:
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + time_)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/intention_discovery.json"

            encoded_img_list = []
            for video_dir in video_dirs:
                all_files = os.listdir(video_dir)
                image_paths = [f for f in all_files if f.endswith(('.jpg', '.png'))]
                for img_path in image_paths:
                    img_vis = cv2.imread(os.path.join(video_dir, img_path))
                    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)  # Uncomment if needed
                    encoded_img = encode_image(img_vis)
                    encoded_img_list.append(encoded_img)

            print("=" * 50)
            print("=" * 20, "Discovering Intention", "=" * 20)
            print("=" * 50)
            
            json_data = query(system, [(intention_user_contents_filled, encoded_img_list)], [], save_path, model_dict['intention_discovery'], temperature_dict['intention_discovery'], debug=False)
    
        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            intention_response = json_data["res"]
            print(intention_response)
            print()
    
    else:
        # Use local Llama inference
        if existing_response is None:
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + time_)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/intention_discovery.json"

            print("=" * 50)
            print("=" * 20, "Discovering Intention", "=" * 20)
            print("=" * 50)

            # Prepare image paths
            image_paths = None
            if video_dirs:
                video_dir = video_dirs[0]
                all_files = os.listdir(video_dir)
                image_paths = [os.path.join(video_dir, f) for f in all_files if f.endswith(('.jpg', '.png'))]
            
            # Get temperature from dict if available
            temp = temperature_dict.get('intention_discovery', 0.2) if temperature_dict else 0.2
            
            # Call the local inference function
            json_data = llm_local_inference(
                user_content=intention_user_contents_filled,
                image_paths=image_paths,
                save_path=save_path,
                temperature=temp,
                max_tokens=4096
            )

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            intention_response = json_data["res"]
            print(intention_response)
            print()


    return intention_user_contents_filled, json_data["res"] 