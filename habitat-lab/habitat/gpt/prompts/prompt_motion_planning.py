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
    1.	A human motion list: {motion_sets_list}.
    2.  Proposed activities with broken down predicates across a day.
    3.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    4.	A dict mapping rigid, dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.

    As a human in the house, plan your motion to accomplish each predicate.

    Constraints:
    1.  ONLY USE MOTION FROM HUMAN MOTION LIST AND OBJECT FROM STATIC AND DYNAMIC OBJECT DICTS. Do not introduce imaginary motion or objects!
    2.  Use four types of motion: walk, walk and pick, walk and place, and customized motion from the human motion list.
    3.  Start with no object in hand. Objects picked must be placed before picking another.
    4.  Before interacting with an object using customized motion, walk towards it.
    5.  Customized motion can be conducted with picked object in hand.
    6.  Do not pick and place an object back at its original position. Focus on rearranging/organizing according to the room layout.
    7.  Only dynamic objects can be picked/placed. Static objects can be interacted with using customized motions.
    8. 	All objects are rigid and cannot deform, disassemble, or transform.

    Format:
    1. walk: (walk, obj_name, obj_id, type) # move towards obj_id. type: static/dynamic
    2. walk and pick: (walk_pick, obj_name, obj_id, type) # move towards and pick obj_id
    3. walk and place: (walk_place, obj_name, obj_id, type) # move towards and place the object at obj_id's location
    4. customized motion: (motion, customized_motion) # use a motion from the human motion list

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
        
        json_data = query(system, [("", []), ("", []), (motion_user_contents_filled, [])], [("", []), (conversation_hist[1][1], [])], save_path, model_dict['motion_planning'], temperature_dict['motion_planning'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        motion_response = json_data["res"]
        print(motion_response)
        print()
        
    return motion_user_contents_filled, json_data["res"] 
