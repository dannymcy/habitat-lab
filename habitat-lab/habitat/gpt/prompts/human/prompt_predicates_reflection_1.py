import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def reflect_predicates_prompt_1(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory):
    contents = f"""
    Input:
    1.  The proposed intention at time: {time_}.
    2.	A dict mapping rigid, static objects to their IDs and rooms: {obj_room_mapping[0]}.
    3.  Your Big Five scores: {profile_string[1]} (scale 0-5) and human profile: {profile_string[0]}
    4.  Most relevant human intentions proposed at previous times: {retrieved_memory[0]} (if empty, ignore it—this means it's the first intention of the day).
    5.  Most relevant human tasks proposed at previous times.ids: {retrieved_memory[1]} (if empty, ignore it—this means it's the first intention of the day).

    Your task is to check if the temporal dependence and human profile are strictly followed in each task, and revise to make better if necessary.

    Instructions:
    1.  Tasks should be continuous and logical, and align with your Big 5 scores and profile.
    2.  Tasks must have temporal dependence with the previous intentions and tasks, with detailed explanation mentioning previous intentions and tasks explicitly.
    3.  For interacting with fixed, static objects, use only objects from the given static object dict. For objects in hand, a robot will provide them.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    Reflect Each Task: 
    1. no mistake or change made.
    2. ...
    Revised Tasks: 
    1. Thought: detailed descriptions of the task. Reason_human: why it alignes with your Big 5 scores and profile. Reason_intentions: how it depends on previous, relevant intentions at [list of time]. Reason_tasks: how it depends on previous, relevant tasks at [list of time.id]. Act: [type: 1, inter_obj_id: real int, inter_obj_name: xxx, inhand_obj_name: yyy, motion: free-form motion]
    2. ...
    """
    return contents


def reflect_predicates_1(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    predicates_user_contents_filled = reflect_predicates_prompt_1(time_, sampled_motion_list, obj_room_mapping, profile_string, retrieved_memory)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/predicates_reflection_1.json"

        print("=" * 50)
        print("=" * 20, "Reflecting Tasks Dependence", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), ("", []), (predicates_user_contents_filled, [])], [("", []), (conversation_hist[1][1], [])], save_path, model_dict['predicates_reflection'], temperature_dict['predicates_reflection'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        predicates_response = json_data["res"]
        print(predicates_response)
        print()
        
    return predicates_user_contents_filled, json_data["res"] 
