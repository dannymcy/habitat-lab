import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def reflect_predicates_prompt_2(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory):
    contents = f"""
    Input:
    1.  The proposed activity at time: {time_}.
    2.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.

    Your task is to check if the instructions are strictly followed in each predicate, and revise to make better if necessary.

    Instructions:
    1.  Break down the activity into 5 predicates for collaboration with a robot.
    2.	Predicate types: 
        - Type 1: Creative, reasonable free-form human motion interacting or approaching a fixed, static object (static objects cannot be moved) with an object in hand provided by the robot (e.g., sit on sofa with TV remote control in hand, wipe table with tissue in hand, squat with dumbbell in hand near rug).
    3.  For interacting with fixed, static objects, use only objects from the given static object dict (exact name). For objects in hand, a robot will provide them.
    4   Both interacting and inhand objects must be specified. Importantly, they cannot be none.
    5.  Free-form motion should be diverse. Examples: {sampled_motion_list}. Feel free to propose others.
    6. 	All objects are rigid and cannot deform, disassemble, or transform.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Reflect Each Predicates: 
    1. no mistake or change made.
    2. ...
    Revised Predicates: 
    1. Thought: detailed descriptions of the predicate. Reason_human: why it alignes with your profile. Reason_activities: how it depends on previous, relevant activities at [list of time]. Reason_predicates: how it depends on previous, relevant predicates at [list of time.id]. Act: [type: 1, inter_obj_id: real int, inter_obj_name: xxx, inhand_obj_name: yyy, motion: free-form motion]
    2. ...
    """
    return contents


def reflect_predicates_2(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    predicates_user_contents_filled = reflect_predicates_prompt_2(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/predicates_reflection.json"

        print("=" * 50)
        print("=" * 20, "Reflecting Predicates Mistakes", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), ("", []), ("", []), (predicates_user_contents_filled, [])], [("", []), ("", []), (conversation_hist[2][1], [])], save_path, model_dict['predicates_reflection'], temperature_dict['predicates_reflection'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        predicates_response = json_data["res"]
        print(predicates_response)
        print()
        
    return predicates_user_contents_filled, json_data["res"] 
