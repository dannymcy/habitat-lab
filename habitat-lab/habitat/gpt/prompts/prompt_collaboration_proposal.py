import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def propose_collaboration_prompt(time_, sampled_motion_list, intention, predicate, thought, act):
    contents = f"""
    Input:
    1.  Human intention: {intention} at time: {time_}.
    2.  Current human activity: {predicate}.
    2.  Robot assistance to enhance comfort, its thought: {thought}.
    3.  Robot's objects from its magical box to give you: {act}.

    You are the human. You decide if the robot's objects make you more comfortable.
    Instructions:
    1.  For each robot thought and object provided, decide if you want to use the object. 
    2.	If you accept the object, propose a free-form human motion with it. Examples: {sampled_motion_list}. Feel free to propose others.
    3. 	All objects are rigid and cannot deform, disassemble, or transform.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    collaboration: 
    1. Thought: basic descriptions. Accepted: yes/no. Act: [motion: free-form motion]/none if not accepted
    2. ...
    """
    return contents


def propose_collaboration(time_, sampled_motion_list, extracted_planning, predicate, thought, act, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    intention = extracted_planning[f"Time: {time_}"]["Intention"]
    collaboration_user_contents_filled = propose_collaboration_prompt(time_, sampled_motion_list, intention, predicate, thought, act)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/collaboration_proposal.json"

        print("=" * 50)
        print("=" * 20, "Proposing Collaboration", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (collaboration_user_contents_filled, [])], [("", [])], save_path, model_dict['collaboration_proposal'], temperature_dict['collaboration_proposal'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        collaboration_response = json_data["res"]
        print(collaboration_response)
        print()
        
    return collaboration_user_contents_filled, json_data["res"] 
