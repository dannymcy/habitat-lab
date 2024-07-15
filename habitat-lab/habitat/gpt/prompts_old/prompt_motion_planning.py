import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def generate_motion_planning(motion_sets_list, obj_room_mapping):
    contents = f"""
    Input:
    1.  Proposed activities with broken down predicates across a day.
    2.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    3.	A dict mapping rigid, dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.

    As a human in the house, the predicates are planned to do the activity at 9pm. Your task is to plan the motion to conduct each prediate by adding movements and hand actions.

    Constraints:
    1.  Use four types of motion: walk, walk and pick, walk and place, and customized motion in the predicate.
    2.  Start with no object in hand. Objects picked must be placed before picking another.
    3.  Before interacting with an object using customized motion, walk towards it.
    4.  Customized motion can be conducted with picked object in hand.
    5   walk_pick and walk_place come in pairs.
    6.  Do not pick and place an object back at its original position. Focus on rearranging/organizing according to the room layout.

    Format:
    1. walk: (walk, obj_name, obj_id, type) # move towards obj_id. type: static/dynamic
    2. walk and pick: (walk_pick, obj_name, obj_id, type) # move towards and pick obj_id
    3. walk and place: (walk_place, obj_name, obj_id, type) # move towards and place the object at obj_id's location
    4. customized motion: (motion, customized_motion) # use the motion in the predicate (exact wording)

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Predicates: 
    1. obj_id: real int. obj_name: xxx. type: static/dynamic. basic descriptions. 
    2. ...
    Planning (corresponding to each predicate):
    1. [(walk_pick, obj_name, obj_id, type), ...]
    2. ...
    """
    return contents


def plan_motion(motion_sets_list, obj_room_mapping, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    motion_user_contents_filled = generate_motion_planning(motion_sets_list, obj_room_mapping)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/motion_planning.json"

        print("=" * 50)
        print("=" * 20, "Planning Motion", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), ("", []), ("", []), (motion_user_contents_filled, [])], [("", []), ("", []), (conversation_hist[2][1], [])], save_path, model_dict['motion_planning'], temperature_dict['motion_planning'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        motion_response = json_data["res"]
        print(motion_response)
        print()
        
    return motion_user_contents_filled, json_data["res"] 
