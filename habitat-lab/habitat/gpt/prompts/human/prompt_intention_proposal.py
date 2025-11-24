import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query, llm_local_inference


def propose_intention_prompt(time_, room_list, profile_string, retrieved_memory):
    contents = f"""
    Input:
    1.  Current time: {time_}.
    2.	A list of rooms in the house (ignore small spaces like closets): {room_list}.
    3.  Your Big Five scores: {profile_string[1]} (scale 1-5) and human profile: {profile_string[0]}
    4.  Most relevant human intentions proposed at previous times: {retrieved_memory[0]} (ignore if empty—this means it's the first intention of the day).
    5.  Most relevant human tasks proposed at previous times.ids: {retrieved_memory[1]} (ignore if empty—this means it's the first task of the day).

    You are a human living in the house. Propose your intention at current time.

    Instructions:
    1.	Intention must align with your Big 5 scores and reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects.
    2.	Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
    3.  Intention must have temporal dependence but be non-repetitive with the previous intentions and tasks.
    4.  Intention should be within the house.
    5. 	All objects are rigid and cannot deform, disassemble, or transform.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm (e.g, 9 am)
    Intention: basic descriptions.
    Reason_human: detailed descriptions of why it follows your Big 5 scores and profile.
    Reason_intentions: detailed descriptions of why it has temporal dependence with the previous, relevant intentions at [list of time]. 
    Reason_tasks: detailed descriptions of why it has temporal dependence with the previous, relevant tasks at [list of time.id].
    """
    return contents


def propose_intention(time_, room_list, profile_string, retrieved_memory, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, gpt=True):

    intention_user_contents_filled = propose_intention_prompt(time_, room_list, profile_string, retrieved_memory)

    if gpt: 
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + time_)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/intention_proposal.json"

            print("=" * 50)
            print("=" * 20, "Proposing Intention", "=" * 20)
            print("=" * 50)
            
            json_data = query(system, [(intention_user_contents_filled, [])], [], save_path, model_dict['intention_proposal'], temperature_dict['intention_proposal'], debug=False)
    
        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            intention_response = json_data["res"]
            print(intention_response)
            print()
    
    else:
        # Use local Llama inference
        if existing_response is None:
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + time_)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/intention_proposal.json"

            print("=" * 50)
            print("=" * 20, "Proposing Intention", "=" * 20)
            print("=" * 50)

            # Get temperature from dict if available
            temp = temperature_dict.get('intention_proposal', 0.7) if temperature_dict else 0.7
            
            # Call the local inference function (no images needed for intention proposal)
            json_data = llm_local_inference(
                user_content=intention_user_contents_filled,
                image_paths=None,  # No images needed for this function
                save_path=save_path,
                temperature=temp,
                max_tokens=4096
            )

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            intention_response = json_data["res"]
            print(intention_response)
            print()
            
    return intention_user_contents_filled, json_data["res"] 
