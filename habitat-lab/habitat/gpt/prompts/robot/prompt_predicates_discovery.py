import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def discover_predicates_prompt(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping):
    contents = f"""
    Input:
    1.  Human intention: {intention} at time: {time_}.
    2.  A dict mapping rigid, static furnitures to their IDs and rooms: {obj_room_mapping[0]}.
    3.  Inferred Big Five personality scores: {fuzzy_traits} (ignore if empty—this means it's your first collaboration with this human).
    4.  Most relevant human intentions discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).
    5.  Most relevant human tasks discovered at previous times.ids: {retrieved_memory[1]} (ignore if empty—this means it's the first task of the day).

    You are a robot assisting a human.

    Instructions:
    1.  Deduce the human's intention and break down into 5 tasks.
    2.	Task type: For each human task, provide small, handable objects from a magical box. Furnitures in the dict are for room understanding and cannot be used.
    3.  For each task, propose if you should confirm it by specifying a confidence score (0 to 1).
    4. 	All objects are rigid and cannot deform, disassemble, or transform.
    5.  Use Big 5 scores, and temporal dependence based on previous intentions and tasks as hints.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Tasks: 
    1. Thought: basic descriptions. Confidence: yyy. Reason: reasons of the proposed confidence with respect to Big 5 scores and temporal depandence. Act: [obj_name: xxx]
    2. ...
    """
    return contents


def discover_predicates(time_, retrieved_memory, fuzzy_traits, obj_room_mapping, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    intention = extract_intentions(conversation_hist[0][1])[0]
    predicates_user_contents_filled = discover_predicates_prompt(time_, intention, retrieved_memory, fuzzy_traits, obj_room_mapping)

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
        
        json_data = query(system, [("", []), (predicates_user_contents_filled, [])], [("", [])], save_path, model_dict['predicates_discovery'], temperature_dict['predicates_discovery'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        predicates_response = json_data["res"]
        print(predicates_response)
        print()
        
    return predicates_user_contents_filled, json_data["res"] 
