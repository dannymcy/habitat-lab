import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def propose_predicates_prompt(time_, sampled_motion_list, obj_room_mapping):
    contents = f"""
    Input:
    1.  The proposed activity at time: {time_}.
    2.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    3.	A dict mapping rigid, dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.

    You are a human living in the house.

    Instructions:
    1.  Break down the activity into several (2 to 5) predicates, tracking any picked object in hand at each step.
    2.	Predicate types:
        - Type 1: Creative, reasonable free-form human motion interacting with a fixed, static object (static objects cannot be moved)
        - Type 2: Pick a dynamic object
        - Type 3: Place the picked dynamic object at the place of the target object
    3.  Use only objects from the given static and dynamic object dicts.
    4.  Predicates should be continuous and logical.
    5.  Start with no object in hand. Objects picked must be placed before picking another.
    6.  Free-form motion can be conducted with picked object in hand. Type 1 predicate should be diverse and as majority of the predicates. Examples: {sampled_motion_list}. Feel free to propose others.
    7. 	All objects are rigid and cannot deform, disassemble, or transform.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Predicates: 
    1. Thought: basic descriptions. Act: [type: 1/2/3, obj_id: real int, obj_name: xxx, property: static/dynamic, motion: free-form motion/pick/place]
    2. ...
    Tracking: none or obj_id (real int) in a list, with length equal to the number of predicates.

    Examples:
    Predicates: 
    1. Thought: Pick up the cup. Act: [type: 2, obj_id: cup_id (real int), obj_name: cup, property: dynamic, motion: pick]
    2. Thought: Drink the water in the cup. Act: [type: 1, obj_id: cup_id (real int), obj_name: cup, property: dynamic, motion: drink_water]
    3. Thought: Place the cup on the table. Act: [type: 3, obj_id: table_id (real int), obj_name: table, property: static, motion: place]
    Tracking: [cup_id, cup_id, none]
    """
    return contents


def propose_predicates(time_, sampled_motion_list, obj_room_mapping, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    predicates_user_contents_filled = propose_predicates_prompt(time_, sampled_motion_list, obj_room_mapping)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/predicates_proposal.json"

        print("=" * 50)
        print("=" * 20, "Proposing Predicates", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (predicates_user_contents_filled, [])], [(conversation_hist[0][1], [])], save_path, model_dict['predicates_proposal'], temperature_dict['predicates_proposal'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        predicates_response = json_data["res"]
        print(predicates_response)
        print()
        
    return predicates_user_contents_filled, json_data["res"] 
