import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def approve_collaboration_prompt(time_, intentions, human_thoughts, human_acts, robot_thoughts, robot_acts):
    contents = f"""
    Input:
    1.  Human intention: {intentions[0]} at time: {time_}.
    2.  Robot inferred intention {intentions[1]}.
    3.  Five current human tasks: {human_thoughts} and desired objects: {human_acts}.
    4.  Robot's thoughts on enhancing comfort: {robot_thoughts} and offered objects: {robot_acts}.

    You are the human. Decide if the robot's assistance and offered objects align with your needs.
    
    Instructions:
    1.  Determine if the robot's inferred intention aligns with your intention.
    2.  For each robot thought and object, decide if you want to use the object. Consider that a good thought must be paired with a reasonable object to meet your needs.
    3.  Respond with yes/no for each, followed by an explanation.

    Write in the following format. Do not output anything else:
    Intention: yes/no
    Tasks: [yes, no, ...]
    Reasons_intention: xxx
    Reasons_tasks:
    1. ...
    2. ...
    """
    return contents


def approve_collaboration(time_, intentions, human_thoughts, human_acts, robot_thoughts, robot_acts, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    collaboration_user_contents_filled = approve_collaboration_prompt(time_, intentions, human_thoughts, human_acts, robot_thoughts, robot_acts)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/collaboration_approval.json"

        print("=" * 50)
        print("=" * 20, "Approving Collaboration", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (collaboration_user_contents_filled, [])], [("", [])], save_path, model_dict['collaboration_approval'], temperature_dict['collaboration_approval'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        collaboration_response = json_data["res"]
        print(collaboration_response)
        print()
        
    return collaboration_user_contents_filled, json_data["res"] 
