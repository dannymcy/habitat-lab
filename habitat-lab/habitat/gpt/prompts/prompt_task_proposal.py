import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def propose_task_prompt(obj_idx):
    contents = f"""
    Input:
    1.	RGB image (640, 480) = (width, height) with multiple objects.
    2.	Detected objects with index.

    You are a single robot hand working in this image scene to perform simple household tasks. Tasks must be discovered from the image. Consider objects’ affordances and feel free to make assumptions (e.g., a bowl can contain water) and interactions with other objects (e.g., pouring water from a cup into a bowl). 

    Task types:
    1.	Interaction between the manipulating object and one of the detected objects (involve translation, or translation + rotation).
    2.	Rotate manipulating object (involve rotation). 

    Strictly follow constraints:
    1.	Exclude tasks involving assembly or disassembly of objects.
    2.	Exclude tasks involving cleaning or functionality testing. 
    3.	Exclude tasks involving imaginary objects.
    4. 	Manipulating object moves; interacting object static.
    5. 	Assume all objects are rigid, without joints or moveable parts (i.e., cannot deform, disassemble, transform). This applies even to objects that are typically articulated (e.g., laptop).

    Propose 3 tasks (2 interaction, 1 rotation) for manipulating Object {obj_idx}. Write in the following format. Do not output anything else:
    Task name: xxx
    Manipulating obj idx: {obj_idx}
    Interacting obj idx: obj_idx (actual integer, or manipulating obj idx)
    Description: basic descriptions.
    """
    return contents


def propose_task(img_vis, binding_box_vis, obj_idx, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    task_user_contents_filled = propose_task_prompt(obj_idx)
    encoded_img = encode_image(img_vis)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / ("obj_" + str(obj_idx) + "_" + time_string)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/task_proposal.json"

        print("=" * 50)
        print("=" * 20, "Proposing Tasks", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (task_user_contents_filled, encoded_img)], [(conversation_hist[0][1], [])], save_path, model_dict['task_proposal'], temperature_dict['task_proposal'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        task_response = json_data["res"]
        print(task_response)
        print()
        
    return task_user_contents_filled, json_data["res"] 
