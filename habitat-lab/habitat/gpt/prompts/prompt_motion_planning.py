import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def generate_motion_planning(obj_room_mapping):
    contents = f"""
    Input:
    1.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    2.	A dict mapping rigid, dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.
    3.  The proposed activities across a day.

    You are a human living in the house. Break down your activities across a day into concrete predicates.

    Constraints:
    1.  Break down each activity into several predicates.
    2.  Predicates should be continuous and logical, if possible.
    2.	Each predicate involves one object.
    3.  Only dynamic objects can be moved, but you can interact with fixed, static objects.
    4. 	All objects are rigid and cannot deform, disassemble, or transform.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Predicates: 
    1. obj_id: real int. obj_name: xxx. type: static/dynamic. basic descriptions. 
    2. ...
    """
    return contents


def plan_motion(obj_room_mapping, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    predicates_user_contents_filled = propose_predicates_prompt(obj_room_mapping)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
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
