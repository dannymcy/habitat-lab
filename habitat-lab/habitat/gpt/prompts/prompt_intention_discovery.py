import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query
import cv2


def discover_intention_prompt(time_):
    contents = f"""
    Input:
    1.	Sequence of images showing human motion.
    2.  Time of the day: {time_}

    You are a robot assisting a human. Identify the human's intention and specific activity.

    Instructions:
    1.	Intention must be high-level and human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
    2.  You only see one human motion. Map this motion to a higher-level intention without mentioning the specific motion.
    3. 	Use the time of day as a hint (e.g., at 12 pm, it is likely lunch time).
    4.  Based on how confident you think your observed human intention is, propose if you want to ask the human to confirm the intention (yes/no).

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Confidence: yes/no
    Intention: basic descriptions (e.g., Prepare for bed and unwind in the bedroom.).
    Activity: basic descriptions.
    """
    return contents


def discover_intention(time_, video_dir, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    intention_user_contents_filled = discover_intention_prompt(time_)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/intention_discovery.json"

        encoded_img_list = []
        all_files = os.listdir(video_dir)
        image_paths = [f for f in all_files if f.endswith(('.jpg', '.png'))]
        for img_path in image_paths:
            img_vis = cv2.imread(os.path.join(video_dir, img_path))
            # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            encoded_img = encode_image(img_vis)
            encoded_img_list.append(encoded_img)

        print("=" * 50)
        print("=" * 20, "Discovering Intention", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [(intention_user_contents_filled, encoded_img_list)], [], save_path, model_dict['intention_discovery'], temperature_dict['intention_discovery'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        intention_response = json_data["res"]
        print(intention_response)
        print()
        
    return intention_user_contents_filled, json_data["res"] 
