import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query, llm_local_inference


def discover_predicates_prompt_1(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping):
    contents = f"""
    Input:
    1.  Human intention: {intention} at time: {time_}.
    2.  First human task: {retrieved_memory[2]} of the intention.
    3.  A dict mapping static (fixed) furnitures to their IDs and rooms: {obj_room_mapping[0]}.
    4.  A dict mapping dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.
    5.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    6.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

    You are a robot assisting a human.

    Instructions:
    1.  Infer 2 additional tasks based on the human intention and initial task.
    2.	Task type: For each task, pick a dynamic object and place on a fixed, static object (static objects cannot be moved).
    3.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    4.  Tasks must have temporal dependence with the intentions at previous times.
    5.  Ensure each task includes Act: [static_obj_name: your chosen static obj, dynamic_obj_name: your chosen dynamic obj] (exact format).

    Write in the following format (1 given human task + 2 inferred by you). Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: detailed descriptions of the task. Reason_human: why it alignes with your Big 5 scores and profile. Reason_intentions: how it depends on previous, relevant intentions. Act: [static_obj_name: xxx, dynamic_obj_name: yyy]
    2. ...
    """
    return contents


def discover_predicates_prompt_2(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping):
    contents = f"""
    Input:
    1.  Human intention: {intention} at time: {time_}.
    2.  A dict mapping rigid, static furnitures to their IDs and rooms: {obj_room_mapping[0]}.
    3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    4.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

    You are a robot assisting a human.

    Instructions:
    1.  Break down the intention into 5 tasks.
    2.	Task type: For each human task, provide one small, handable object from a magical box. Furnitures in the dict are for room understanding and cannot be used.
    3.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    4.  Tasks must have temporal dependence with the intentions at previous times.
    5. 	All objects are rigid and cannot deform, disassemble, or transform.
    6.  Ensure each task includes Act: [obj_name: your chosen object] (exact format).

    Write in the following format. Make sure there are 5 tasks. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: detailed descriptions of the task. Reason_human: why it alignes with your Big 5 scores and profile. Reason_intentions: how it depends on previous, relevant intentions. Act: [obj_name: xxx]
    2. ...
    """
    return contents


def discover_predicates_finetuning_response(time_, intention, thoughts, acts, predicates_num):
    contents = f"Time: {time_}\nIntention: {intention}\nTasks:\n"

    # Loop over the number of predicates and append each Thought-Act pair
    for i in range(predicates_num):
        contents += f"{i+1}. Thought: {thoughts[i]} Act: [obj_name: {acts[i]}]\n"
    
    return contents


def discover_predicates_finetuning_prompt_1(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping):
    contents = f"""
    Input:
    1.  Human intention: {intention} at time: {time_}.
    2.  First human task: {retrieved_memory[2]} of the intention.
    3.  A dict mapping static (fixed) furnitures to their IDs and rooms: {obj_room_mapping[0]}.
    4.  A dict mapping dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.
    5.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    6.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

    You are a robot assisting a human.

    Instructions:
    1.  Infer 2 additional tasks based on the human intention and initial task.
    2.	Task type: For each task, pick a dynamic object and place on a fixed, static object (static objects cannot be moved).
    3.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    4.  Tasks must have temporal dependence with the intentions at previous times.
    5.  Ensure each task includes Act: [static_obj_name: your chosen static obj, dynamic_obj_name: your chosen dynamic obj] (exact format).

    Write in the following format (make sure there are 3 tasks: 1 given human task + 2 inferred by you). Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: detailed descriptions of the task. Act: [static_obj_name: xxx, dynamic_obj_name: yyy]
    2. ...
    """
    return contents


def discover_predicates_finetuning_prompt_2(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping):
    contents = f"""
    Input:
    1.  Human intention: {intention} at time: {time_}.
    2.  A dict mapping rigid, static furnitures to their IDs and rooms: {obj_room_mapping[0]}.
    3.  Inferred Big Five personality scores: {fuzzy_traits[1]} (ignore if empty—this means it's your first collaboration with this human).
    4.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).

    You are a robot assisting a human.

    Instructions:
    1.  Break down the intention into 5 tasks.
    2.	Task type: For each human task, provide one small, handable object from a magical box. Furnitures in the dict are for room understanding and cannot be used.
    3.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    4.  Tasks must have temporal dependence with the intentions at previous times.
    5. 	All objects are rigid and cannot deform, disassemble, or transform.
    6.  Ensure each task includes Act: [obj_name: your chosen object] (exact format).

    Write in the following format. Make sure there are 5 tasks. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: detailed descriptions of the task. Act: [obj_name: xxx]
    2. ...
    """
    return contents


def discover_predicates(time_, retrieved_memory, intention, fuzzy_traits, obj_room_mapping, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, method="main", collab=2, gpt=True):

    # intention = extract_intentions(conversation_hist[0][1])[0]
    if collab == 1:
        if method == "finetuning":
            predicates_user_contents_filled = discover_predicates_finetuning_prompt_1(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping)
        elif method == "main":
            predicates_user_contents_filled = discover_predicates_prompt_1(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping)
    elif collab == 2:
        if method == "finetuning":
            predicates_user_contents_filled = discover_predicates_finetuning_prompt_2(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping)
        elif method == "main":
            predicates_user_contents_filled = discover_predicates_prompt_2(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping)

    if gpt or method == "finetuning":
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + time_)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/predicates_discovery.json"

            print("=" * 50)
            print("=" * 20, "Discovering Tasks", "=" * 20)
            print("=" * 50)
            
            if method == "finetuning":
                json_data = query(system, [("", []), (predicates_user_contents_filled, [])], [("", [])], save_path, model_dict['finetuning'], temperature_dict['finetuning'], debug=False)
                # print()
                # print(model_dict['finetuning'])
                # print()
            elif method == "main":
                json_data = query(system, [("", []), (predicates_user_contents_filled, [])], [("", [])], save_path, model_dict['predicates_discovery'], temperature_dict['predicates_discovery'], debug=False)
    
        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            predicates_response = json_data["res"]
            print(predicates_response)
            print()
    
    else:
        # Use local Llama inference
        if existing_response is None:
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + time_)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/predicates_discovery.json"

            print("=" * 50)
            print("=" * 20, "Discovering Tasks", "=" * 20)
            print("=" * 50)

            # Get temperature from dict if available
            temp = temperature_dict.get('predicates_discovery', 0.2) if temperature_dict else 0.2
            
            # Call the local inference function (no images needed for predicates discovery)
            json_data = llm_local_inference(
                user_content=predicates_user_contents_filled,
                image_paths=None,  # No images needed for this function
                save_path=save_path,
                temperature=temp,
                max_tokens=4096
            )

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            predicates_response = json_data["res"]
            print(predicates_response)
            print()


    return predicates_user_contents_filled, json_data["res"] 
