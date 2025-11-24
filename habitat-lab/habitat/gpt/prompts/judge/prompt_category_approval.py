import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from habitat.gpt.prompts.utils import *
from habitat.gpt.query import query


def approve_category_prompt_1(time_, human_thoughts, human_acts, robot_thoughts, robot_acts, predicates_num):
    contents = f"""
    Input:
    1.  Human actions ({predicates_num} items) - pick and place tasks: {human_acts}
    2.  Robot actions ({predicates_num} items) - pick and place tasks: {robot_acts}
    
    Instructions:
    1.  For each robot action, determine if it accomplishes the **same semantic goal** as the corresponding human action by comparing:
       - Whether the picked objects belong to the same category (e.g., fruits, tools, containers)
       - Whether the placement locations serve similar purposes (e.g., tables, storage areas, workspaces)
       
       Use broad, semantic categories for objects. Examples include:
       - apple, banana, grapes, strawberry → **fruits**
       - broccoli, carrot, spinach, lettuce → **vegetables**
       - knife, fork, spoon, chopsticks → **cutlery / eating utensils**
       - cup, mug, glass, bottle, thermos → **drinkware / containers**
       - laptop, tablet, smartphone, charger → **electronics / digital devices**
       - chair, sofa, stool, bench → **seating / furniture**
       - soap, shampoo, towel, toothbrush → **hygiene / personal care**
       - book, notebook, pen, folder → **stationery / study materials**
       - pan, pot, spatula, cutting board → **kitchenware / cooking tools**
       
       For placement locations, consider functional equivalence:
       - Different tables/desks → **work surfaces**
       - Different shelves/cabinets → **storage areas**
       - Different counters/benches → **preparation areas**
       
    2.  Consider each robot action individually against all human actions. Approve it if it accomplishes a similar goal to ANY human action (sequence does not matter).
    3.  Be fair in judgment—an action is approved if both the object category AND placement purpose align semantically, even if specific objects or locations differ.
    4.  Respond with yes/no for each, followed by an explanation. Ensure there are {predicates_num} items in the "Tasks" list.

    Write in the following format. Do not output anything else:
    Tasks: [yes, no, ...]
    Reasons_tasks:
    1. ...
    2. ...
    """
    return contents


def approve_category_prompt_2(time_, human_thoughts, human_acts, robot_thoughts, robot_acts, predicates_num):
    contents = f"""
    Input:
    1.  Human-desired objects ({predicates_num} items) for some tasks: {human_acts}
    2.  Robot-offered objects ({predicates_num} items) for those tasks: {robot_acts}
    
    Instructions:
    1.  For each robot-offered object, decide whether it belongs to the **same object category** as the corresponding human-desired object. Use broad, semantic object categories. Examples include (but are not limited to):
       - apple, banana, grapes, strawberry → **fruits**
       - broccoli, carrot, spinach, lettuce → **vegetables**
       - knife, fork, spoon, chopsticks → **cutlery / eating utensils**
       - cup, mug, glass, bottle, thermos → **drinkware / containers**
       - laptop, tablet, smartphone, charger, USB cable → **electronics / digital devices**
       - pillow, blanket, quilt, bedsheet, mattress topper → **bedding / sleep-related items**
       - chair, sofa, stool, bench, armchair → **seating / furniture**
       - T-shirt, sweater, jacket, pants, socks → **clothing / apparel**
       - soap, shampoo, towel, toothbrush, toothpaste → **hygiene / personal care**
       - book, notebook, pen, folder, highlighter → **stationery / study materials**
       - pan, pot, spatula, cutting board, whisk → **kitchenware / cooking tools**
       - toy car, puzzle, plush toy, LEGO, action figure → **toys / entertainment objects**
       These examples are intentionally broad. Use similar open-category reasoning.
    2.  Consider each robot thought and object individually against the human tasks. Approve it if it meets any one of the human tasks; sequence does not matter.
    3.  Be fair in your judgment—avoid being too generous or too harsh.
    4.  Respond with yes/no for each, followed by an explanation. Ensure there are {predicates_num} items in the "Tasks" list.

    Write in the following format. Do not output anything else:
    Tasks: [yes, no, ...]
    Reasons_tasks:
    1. ...
    2. ...
    """
    return contents


def approve_category(time_, human_thoughts, human_acts, robot_thoughts, robot_acts, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None, collab=2):

    predicates_num = 5 if collab == 2 else 3
    if collab == 1:
        collaboration_user_contents_filled = approve_category_prompt_1(time_, human_thoughts, human_acts, robot_thoughts, robot_acts, predicates_num)
    elif collab == 2:
        collaboration_user_contents_filled = approve_category_prompt_2(time_, human_thoughts, human_acts, robot_thoughts, robot_acts, predicates_num)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / (time_string + "_" + time_)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/category_approval.json"

        print("=" * 50)
        print("=" * 20, "Approving Object Category", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (collaboration_user_contents_filled, [])], [("", [])], save_path, model_dict['collaboration_approval'], temperature_dict['collaboration_approval'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        collaboration_response = json_data["res"]
        print(collaboration_response)
        print()
        
    return collaboration_user_contents_filled, json_data["res"] 

