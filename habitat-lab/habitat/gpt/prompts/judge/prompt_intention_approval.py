import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query
    

def approve_intention_prompt(time_, intentions, intentions_num):
    contents = f"""
    Input:
    1.  Human intention: {intentions[0]} at time: {time_}.
    2.  {intentions_num} robot inferred intentions: {intentions[1]}.

    You are the human. Decide if the robot's intentions align with your intention.
    
    Instructions:
    1.  Determine if the robot's inferred intentions align with your intention.
    2.  Respond with yes/no for each, followed by an explanation. Ensure there are {intentions_num} items in the "Intentions" list.

    Write in the following format. Do not output anything else:
    Intentions: [yes, no, ...]
    Reasons_intentions:
    1. ...
    2. ...
    """
    return contents


def approve_intention_prompting_prompt(time_, intentions):
    contents = f"""
    Input:
    1.  Human intention: {intentions[0]} at time: {time_}.
    2.  Robot inferred intention: {intentions[1]}.

    You are the human. Decide if the robot's intention aligns with your intention.
    
    Instructions:
    1.  Determine if the robot's inferred intention aligns with your intention.
    2.  Respond with [yes]/[no], followed by an explanation. Ensure to put yes/no response in a list.

    Write in the following format. Do not output anything else:
    Intentions: [yes]/[no]
    Reasons_intentions:
    1. ...
    """
    return contents


def approve_intention(time_, intentions, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, method="main", collab=2):

    intentions_num = 5 if collab == 2 else 3
    if method in ["prompting", "finetuning"]:
        collaboration_user_contents_filled = approve_intention_prompting_prompt(time_, intentions)
    elif method in ["main", "ag_human", "random_"]:
        collaboration_user_contents_filled = approve_intention_prompt(time_, intentions, intentions_num)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/intention_approval.json"

        print("=" * 50)
        print("=" * 20, "Approving Intentions", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (collaboration_user_contents_filled, [])], [("", [])], save_path, model_dict['collaboration_approval'], temperature_dict['collaboration_approval'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        collaboration_response = json_data["res"]
        print(collaboration_response)
        print()
        
    return collaboration_user_contents_filled, json_data["res"] 
