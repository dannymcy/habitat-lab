import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query
import cv2


def describe_motion_prompt(time_):
    contents = f"""
    Input:
    1.	Sequence of images showing human motion from robot's and human's perspectives.
    2.  Current time: {time_}.

    Task:
    Describe the human's motion based on the image sequence.

    Instructions:
    1.  Provide a description of the motion frame by frame.
    2.  Incorporate environmental cues to support your description.
    """
    return contents


def describe_motion(time_, video_dirs, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    motion_user_contents_filled = describe_motion_prompt(time_)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/motion_description.json"

        encoded_img_list = []
        for video_dir in video_dirs:
            all_files = os.listdir(video_dir)
            image_paths = [f for f in all_files if f.endswith(('.jpg', '.png'))]
            for img_path in image_paths:
                img_vis = cv2.imread(os.path.join(video_dir, img_path))
                # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)  # Uncomment if needed
                encoded_img = encode_image(img_vis)
                encoded_img_list.append(encoded_img)

        print("=" * 50)
        print("=" * 20, "Describing Motion", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [(motion_user_contents_filled, encoded_img_list)], [], save_path, model_dict["finetuned_base_model"], temperature_dict['finetune'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        motion_response = json_data["res"]
        print(motion_response)
        print()
        
    return motion_user_contents_filled, json_data["res"] 
