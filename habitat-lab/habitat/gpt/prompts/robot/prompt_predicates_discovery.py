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
    1.  Human activity: {intention} at time: {time_}.
    2.  A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    3.  Inferred human traits: {fuzzy_traits} (ignore if empty—this means it's your first collaboration with this human).

    You are a robot assisting a human.
    Instructions:
    1.  Break down the human activity into 5 predicates.
    2.	Predicate type: Deduce the human's activities and provide objects.
    3.  For each predicate, propose if you should confirm it by specifying a confidence score (0 to 1).
    4.  Provide small, handable objects from a magical box. Objects in the dict are for room understanding and cannot be used.
    5. 	All objects are rigid and cannot deform, disassemble, or transform.
    6.  Use inferred human traits (e.g., an athlete likely does morning exercises), and temporal dependence based on previous activities and predicates as hints.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Predicates: 
    1. Thought: basic descriptions. Confidence: yyy. Reason: reasons of the proposed confidence with respect to human traits and temporal depandence. Act: [obj_name: xxx]
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
        print("=" * 20, "Discovering Predicates", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (predicates_user_contents_filled, [])], [("", [])], save_path, model_dict['predicates_discovery'], temperature_dict['predicates_discovery'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        predicates_response = json_data["res"]
        print(predicates_response)
        print()
        
    return predicates_user_contents_filled, json_data["res"] 
