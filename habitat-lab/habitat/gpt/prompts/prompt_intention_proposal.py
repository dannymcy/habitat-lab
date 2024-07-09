import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def propose_intention_prompt(room_list):
    contents = f"""
    Input:
    1.	A list of rooms in the house (ignore small spaces like closets): {room_list}.

    You are an athletic human living in the house. Propose your activities across a day (9am to 9pm) with three-hour intervals.

    Constraints:
    1.	Activities must be diverse yet reasonable based on the house layout and available objects.
    2.	Activities must be high-level and either human-centric (e.g., hygiene, sport, leisure) or room-centric (e.g., clean, organize, set-up). Do not mention specific objects.
    3.  Activities should be non-repetitive.
    4. 	All objects are rigid and cannot deform, disassemble, or transform.

    Write in the following format. Do not output anything else:
    Time: xxx am/pm
    Intention: basic descriptions.
    """
    return contents


def propose_intention(room_list, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    intention_user_contents_filled = propose_intention_prompt(room_list)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
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
        
    return intention_user_contents_filled, json_data["res"] 
