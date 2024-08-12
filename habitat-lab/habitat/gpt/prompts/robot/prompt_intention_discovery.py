import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query
import cv2


def discover_intention_prompt(time_, fuzzy_traits, retrieved_memory):
    contents = f"""
    Input:
    1.	Sequence of images showing human motion from your and human's perspectives.
    2.  Current time: {time_}.
    3.  Inferred human traits: {fuzzy_traits} (ignore if empty—this means it's your first collaboration with this human).
    4.  Most relevant human activities discovered at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first activity of the day).
    5.  Most relevant human predicates discovered at previous times.ids: {retrieved_memory[1]} (ignore if empty—this means it's the first predicate of the day).

    You are a robot assisting a human. Identify the human's intention and propose if you should confirm it by specifying a confidence score.

    Instructions:
    1.	Intention must be high-level and human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
    2.  Map the observed human motion to a higher-level intention without mentioning the specific motion.
    3. 	Use the time of day (e.g., at 12 pm, it is likely lunch time), inferred human traits (e.g., an athlete likely does morning exercises), and temporal dependence based on previous activities and predicates as hints.
    4.  Confidence score should between 0 and 1.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions (e.g., Prepare for bed and unwind in the bedroom).
    Confidence: yyy
    Reason_text: detailed descriptions of why it follows the inferred human traits, has temporal dependence with the previous, relevant activities at [list of time] or predicates at [list of time.id].
    Reason_vis: detailed descriptions with respect to the visual cues.
    """
    return contents


def discover_intention(time_, retrieved_memory, fuzzy_traits, video_dirs, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    intention_user_contents_filled = discover_intention_prompt(time_, fuzzy_traits, retrieved_memory)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/intention_discovery.json"

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
