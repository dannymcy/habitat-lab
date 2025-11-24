import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def approve_predicate_prompt_1(time_, human_thoughts, human_acts, robot_thoughts, robot_acts):
    contents = f"""
    Input:
    1.  Human intention: {human_thoughts[1]} at time: {time_}.
    1.  Three current human tasks: {human_thoughts[0]} at time: {time_}.
    2.  Three robot-inferred tasks on enhancing comfort: {robot_thoughts}.

    You are the human. Decide if the robot's assistance align with your needs.
    
    Instructions:
    1.  Assess if each robot task supports the human tasks and intention. The robot's task doesn't need to be an exact match but should be relevant in purpose, context, or object categories. Use common reasoning to decide if it helps meet your needs.
    2.  Consider each robot thought and object individually against the human tasks. Approve it if it meets any one of the human tasks; sequence does not matter.
    3.  Be fair in your judgment—avoid being too generous or too harsh.
    4.  Respond with yes/no for each, followed by an explanation. Ensure there are 3 items in the "Tasks" list.

    Write in the following format. Do not output anything else:
    Tasks: [yes, no, ...]
    Reasons_tasks:
    1. ...
    2. ...
    """
    return contents


def approve_predicate_prompt_2(time_, human_thoughts, human_acts, robot_thoughts, robot_acts):
    contents = f"""
    Input:
    1.  Five current human tasks: {human_thoughts} and desired objects: {human_acts} at time: {time_}.
    2.  Five robot-inferred tasks on enhancing comfort: {robot_thoughts} and offered objects: {robot_acts}.

    You are the human. Decide if the robot's assistance and offered objects align with your needs.
    
    Instructions:
    1.  For each robot thought and object, decide if you want to use the object. The robot's task doesn't need to be an exact match to the human task, but it should be relevant in purpose or context. Use common reasoning to assess whether the robot's task helps meet your current needs. A good thought must be paired with a reasonable object.
    2.  Consider each robot thought and object individually against the human tasks. Approve it if it meets any one of the human tasks; sequence does not matter.
    3.  Be fair in your judgment—avoid being too generous or too harsh.
    4.  Respond with yes/no for each, followed by an explanation. Ensure there are 5 items in the "Tasks" list.

    Write in the following format. Do not output anything else:
    Tasks: [yes, no, ...]
    Reasons_tasks:
    1. ...
    2. ...
    """
    return contents


def approve_predicate(time_, human_thoughts, human_acts, robot_thoughts, robot_acts, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, collab=2):

    if collab == 1:
        collaboration_user_contents_filled = approve_predicate_prompt_1(time_, human_thoughts, human_acts, robot_thoughts, robot_acts)
    elif collab == 2:
        collaboration_user_contents_filled = approve_predicate_prompt_2(time_, human_thoughts, human_acts, robot_thoughts, robot_acts)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/predicate_approval.json"

        print("=" * 50)
        print("=" * 20, "Approving Predicates", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (collaboration_user_contents_filled, [])], [("", [])], save_path, model_dict['collaboration_approval'], temperature_dict['collaboration_approval'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        collaboration_response = json_data["res"]
        print(collaboration_response)
        print()
        
    return collaboration_user_contents_filled, json_data["res"] 
