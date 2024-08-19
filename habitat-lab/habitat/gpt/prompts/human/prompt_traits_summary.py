import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def summarize_traits_prompt(profile_string):
    contents = f"""
    Input: 
    1. Your Big Five scores: {profile_string[1]} (scale 0-5)
    2. Your human profile: {profile_string}

    Task: Summarize this person's Facebook posts into a first-person self-introduction. Provide a detailed description covering, if presented, the person's job, hobbies, daily activities, food preferences, social life, physical activity, entertainment preferences, travel habits, personal values, goals and aspirations, stress and coping mechanisms, technological use, cultural interests, health and wellness, community involvement, education, financial habits, and personal style. Also, include a comprehensive analysis of the Big Five personality traits, explaining how these scores impact their daily life.

    Write in the following format. Do not output anything else:
    Profile: xxx
    """
    return contents


def summarize_traits(profile_string, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    traits_user_contents_filled = summarize_traits_prompt(profile_string)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/traits_summary.json"

        print("=" * 50)
        print("=" * 20, "Summarizing Traits", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [(traits_user_contents_filled, [])], [], save_path, model_dict['traits_summary'], temperature_dict['traits_summary'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        traits_response = json_data["res"]
        print(traits_response)
        print()
        
    return traits_user_contents_filled, json_data["res"] 
