import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def reflect_predicates_prompt(motion_sets_list, obj_room_mapping):
    contents = f"""
    Input:
    1.	A human motion list: {motion_sets_list}
    2.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    3.	A dict mapping rigid, dynamic objects to their IDs and rooms: {obj_room_mapping[1]}.
    4.  The proposed activities and predicates across a day.

    Your task is to check if the following constraints are strictly followed in each predicate at 9pm, and revise.

    Constraints:
    1.  ONLY USE MOTION HUMAN MOTION LIST (exact wording). Do not introduce imaginary motion!
    2.  ONLY USE OBJECT FROM STATIC AND DYNAMIC OBJECT DICTS (exact wording and ID). Do not introduce imaginary objects!

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Predicates: 
    1. obj_id: real int. obj_name: xxx. type: static/dynamic. motion: yyy. hand_motion: pick/place/none. basic descriptions involving objects and motions.
    2. ...
    """
    return contents


def reflect_predicates(motion_sets_list, obj_room_mapping, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    predicates_user_contents_filled = reflect_predicates_prompt(motion_sets_list, obj_room_mapping)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/predicates_reflection.json"

        print("=" * 50)
        print("=" * 20, "Reflecting Predicates", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), ("", []), (predicates_user_contents_filled, [])], [("", []), (conversation_hist[1][1], [])], save_path, model_dict['predicates_reflection'], temperature_dict['predicates_reflection'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        predicates_response = json_data["res"]
        print(predicates_response)
        print()
        
    return predicates_user_contents_filled, json_data["res"] 
