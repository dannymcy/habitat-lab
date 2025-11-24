import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query, llm_local_inference


def infer_traits_prompt(retrieved_memory, fuzzy_traits):
    contents = f"""
    Input:
    1.  Human intentions at previous times: {retrieved_memory[0]} (ignore if empty—this means it's your first collaboration with this human).
    2.  Human profile: {fuzzy_traits[0]} (ignore if empty—this means it's your first collaboration with this human).

    Task: Mimic this human by:
    1.  Inferring Big Five personality traits (scale 1-5, float) based on the provided intentions.
    2.  Summarizing the human profile (i.e., preferences/habits) based on the intentions within three sentences. Revise the existing human profile if necessary.
    3.  Ensure to follow the exact output format.

    Write in the following format. Do not output anything else:
    Scores: {{'openness': a, 'conscientiousness': b, 'extroversion': c, 'agreeableness': d, 'neuroticism': e}}
    Profile: ...
    Reasons_ocean: explain each ocean.
    Reasons_profile: explain the profile.
    """
    return contents


def infer_traits(time_, retrieved_memory, fuzzy_traits, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, gpt=True):

    traits_user_contents_filled = infer_traits_prompt(retrieved_memory, fuzzy_traits)

    if gpt:
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / time_string
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
    
    else:
        # Use local Llama inference
        if existing_response is None:
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / time_string
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/traits_inference.json"

            print("=" * 50)
            print("=" * 20, "Inferring Traits", "=" * 20)
            print("=" * 50)

            # Get temperature from dict if available
            temp = temperature_dict.get('traits_inference', 0.2) if temperature_dict else 0.2
            
            # Call the local inference function (no images needed for traits inference)
            json_data = llama_local_inference(
                user_content=traits_user_contents_filled,
                image_paths=None,  # No images needed for this function
                save_path=save_path,
                temperature=temp,
                max_tokens=4096
            )

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            traits_response = json_data["res"]
            print(traits_response)
            print()

    
    return traits_user_contents_filled, json_data["res"] 
