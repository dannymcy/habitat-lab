import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def infer_traits_prompt(retrieved_memory, fuzzy_traits):
    contents = f"""
    Input:
    1.  Human intentions at previous times: {retrieved_memory[0]} (ignore if empty—this means it's your first inference).
    2.  Human tasks at previous times.ids: {retrieved_memory[1]} (ignore if empty—this means it's your first inference).

    Task: Infer Big Five personality traits (scale 0-5, float) based on provided intentions and tasks.

    Write in the following format. Do not output anything else:
    Scores: {{'openness': a, 'conscientiousness': b, 'extroversion': c, 'agreeableness': d, 'neuroticism': e}}
    Reasons: explain for each ocean.
    """
    return contents


def infer_traits(time_, retrieved_memory, fuzzy_traits, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    traits_user_contents_filled = infer_traits_prompt(retrieved_memory, fuzzy_traits)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/traits_inference.json"

        print("=" * 50)
        print("=" * 20, "Inferring Traits", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [(traits_user_contents_filled, [])], [], save_path, model_dict['traits_inference'], temperature_dict['traits_inference'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        traits_response = json_data["res"]
        print(traits_response)
        print()
        
    return traits_user_contents_filled, json_data["res"] 
