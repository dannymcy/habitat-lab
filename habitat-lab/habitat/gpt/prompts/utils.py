import cv2
import base64
import numpy as np
import io
import os
import json
import re
import ast
from pathlib import Path


def encode_image(input_img):
    # Check if the image is loaded properly
    if input_img is None:
        raise ValueError("The image could not be loaded. Please check the file path.")
    
    # Encode the image as a JPEG (or PNG) to a memory buffer
    img_vis = input_img.copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode('.png', img_vis)
    if not success:
        raise ValueError("Could not encode the image")

    # Convert the encoded image to bytes and then to a base64 string
    image_bytes = io.BytesIO(encoded_image).read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    # return base64_string
    return f'data:image/png;base64, {base64_string}'


def load_response(prompt_name, prompt_path, file_idx=None, get_latest=True):
    if prompt_path.exists():
        subdirs = [d for d in os.listdir(prompt_path) if os.path.isdir(prompt_path / d)]
        subdirs.sort()
        
        if get_latest and file_idx is None:
            # Find the latest subdirectory
            latest_subdir = max(subdirs, key=lambda d: (prompt_path / d).stat().st_mtime)
            json_file_path = prompt_path / latest_subdir / f"{prompt_name}.json"
            if json_file_path.exists():
                return json_file_path
        elif file_idx is not None:
            selected_subdir = subdirs[file_idx]
            json_file_path = prompt_path / selected_subdir / f"{prompt_name}.json"
            if json_file_path.exists():
                return json_file_path
        else:
            # Process all subdirectories
            responses = []
            for subdir in subdirs:
                json_file_path = prompt_path / subdir / f"{prompt_name}.json"
                if json_file_path.exists():
                    responses.append(json_file_path)
            return responses


def extract_times(conversation_hist):
    """
    Extract the times from the conversation history.
    """
    times = []
    lines = conversation_hist.split("\n")
    for i in range(len(lines)):
        if lines[i].startswith("Time:"):
            time = lines[i].replace("Time:", "").strip()
            times.append(time)
    return times


def extract_intentions(conversation_hist):
    """
    Extract the intention sentences from the conversation history.
    """
    intentions = []
    lines = re.split(r'\n\s*', conversation_hist)  # Split by newlines and ignore leading spaces
    pattern = r"^Intention(?: \d*)?:"
    
    for line in lines:
        line = line.strip()  # Trim leading and trailing spaces
        if re.search(pattern, line):  # Use re.search to find the pattern in the line
            # Remove the "Intention {int}:" or "Intention:" part and strip leading/trailing spaces
            intention = re.sub(pattern, "", line).strip()
            intentions.append(intention)

    return intentions


def parse_act_line(line, collab=2):
    """
    Parse the 'Act' part of a predicate into a list.
    """
    def fallback_extraction(act_content):
        """
        Fallback extraction method to find two integers and their associated strings in the content.
        """
        # Find all integer values in the content
        integers = re.findall(r'\b\d+\b', act_content)
        if len(integers) < 2:
            raise ValueError("Expected at least two integers in the input content.")
        
        # Convert integers to int type
        int_1, int_2 = int(integers[0]), int(integers[1])

        # Extract the text segments between and after the integers, handling potential formatting issues
        pattern = re.compile(rf"{int_1},\s*'(.+?)',\s*{int_2},\s*'(.+?)'")
        match = pattern.search(act_content)

        if match:
            # Properly formatted content
            obj_name_1 = match.group(1).strip()
            obj_name_2 = match.group(2).strip()
        else:
            # Handle cases with misformatting by extracting based on position
            segments = re.split(rf"{int_1}|{int_2}", act_content)
            
            if len(segments) >= 3:
                obj_name_1 = segments[1].strip("', ").strip()
                obj_name_2 = segments[2].strip("', ").strip()
            else:
                raise ValueError("Could not reliably extract object names from the content.")
        
        return [int_1, obj_name_1, int_2, obj_name_2]


    act_start = line.find("Act: [") + len("Act: [")
    act_end = line.find("]", act_start)
    act_content = line[act_start:act_end]
    
    if collab == 1:
        # Try to extract elements in the format [int, str, int, str]
        keys = ["static_obj_id", "static_obj_name", "dynamic_obj_id", "dynamic_obj_name"]
        act_parts_cleaned = []
        
        for key in keys:
            # Find the position of the key and its value
            key_pos = act_content.find(f"{key}: ")
            
            # If the expected format is found, extract by keys
            if key_pos != -1:
                key_pos += len(f"{key}: ")
                next_key_pos = min([act_content.find(f"{next_key}: ") for next_key in keys if f"{next_key}: " in act_content and act_content.find(f"{next_key}: ") > key_pos] + [len(act_content)])
                value = act_content[key_pos:next_key_pos].strip().rstrip(",")
                try:
                    value = int(value)  # Convert value to int if possible
                except ValueError:
                    pass
                act_parts_cleaned.append(value)
            else:
                # If the expected format is not found, attempt to use eval
                try:
                    act_parts_cleaned = eval(act_content)
                except (SyntaxError, ValueError):
                    # If eval fails, use the fallback extraction method
                    act_parts_cleaned = fallback_extraction(act_content)
                break

    elif collab == 2:
        # Define the expected keys in the order they appear
        keys = ["type", "inter_obj_id", "inter_obj_name", "inhand_obj_name", "motion"]
        act_parts_cleaned = []
        
        for key in keys:
            # Find the position of the key and its value
            key_pos = act_content.find(f"{key}: ") + len(f"{key}: ")
            next_key_pos = min([act_content.find(f"{next_key}: ") for next_key in keys if f"{next_key}: " in act_content and act_content.find(f"{next_key}: ") > key_pos] + [len(act_content)])
            value = act_content[key_pos:next_key_pos].strip().rstrip(",")
            
            # Convert value to int if it's a number
            try:
                value = int(value)
            except ValueError:
                pass
            
            act_parts_cleaned.append(value)
    
    return act_parts_cleaned


def parse_thought_line(line):
    """
    Parse the 'Thought' part of a predicate into a list.
    """
    thought_start = line.find("Thought: ") + len("Thought: ")
    act_start = line.find("Act: [", thought_start)
    thought_content = line[thought_start:act_start].strip()
    
    return [thought_content]
    

def extract_code(prompt_name, prompt_path, file_idx, video_path=None, collab=2):
    # Create the video path directory if it doesn't exist
    # video_path.mkdir(parents=True, exist_ok=True)

    if prompt_path.exists():
        json_file_path = load_response(prompt_name, prompt_path, file_idx)
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Initialize the dictionary
        result_dict = {}

        # Extract the "res" field content
        res_content = json_data["res"]
        
        # Split the content by lines
        lines = res_content.split("\n")
        time = lines[0].strip()  # First line is the time
        intention = lines[1].replace("Intention: ", "").strip()  # Second line is the intention
        
        predicate_acts = []
        predicate_thoughts = []
        planning_section = False

        for line in lines[2:]:
            if line.startswith("Revised Tasks:"):
                planning_section = True
                continue
            
            if planning_section:
                if line.strip() != "" and "Thought:" in line:
                    thought_list = parse_thought_line(line.strip())
                    predicate_thoughts.append(thought_list)
                if line.strip() != "" and "Act:" in line:
                    act_list = parse_act_line(line.strip(), collab=collab)
                    predicate_acts.append(act_list)

        result_dict[time] = {
            "Intention": intention,
            "Predicate_Acts": predicate_acts,
            "Predicate_Thoughts": predicate_thoughts
        }

        return result_dict


def extract_thoughts_and_acts(text, search_txt=" Confidence:"):
    """
    Extract the thoughts and acts from the given text.
    """
    lines = text.split('\n')
    thoughts = []
    acts = []

    for line in lines:
        if "Thought:" in line:
            if search_txt == "":
                thought_part = line.split("Thought: ")[1].strip()
            else:
                thought_part = line.split("Thought: ")[1].split(search_txt)[0].strip()
            thoughts.append(thought_part)
            
        if "Act:" in line:
            act_part = line.split("Act: ")[1].strip()
            acts.append(act_part)

    return thoughts, acts


def extract_scores_and_profile(prompt_text):
    # Regular expression to match the Scores dictionary in the prompt
    scores_pattern = r"Scores:\s*\{('openness':\s*[\d.]+,\s*'conscientiousness':\s*[\d.]+,\s*'extroversion':\s*[\d.]+,\s*'agreeableness':\s*[\d.]+,\s*'neuroticism':\s*[\d.]+)\}"
    
    # Regular expression to match the Profile in the prompt
    profile_pattern = r"Profile:\s*(.+)"

    # Search for the scores pattern in the prompt text
    scores_match = re.search(scores_pattern, prompt_text)
    
    # Search for the profile pattern in the prompt text
    profile_match = re.search(profile_pattern, prompt_text)

    if scores_match:
        # Evaluate the dictionary string to convert it into a dictionary object
        scores_str = "{" + scores_match.group(1) + "}"
        scores_dict = eval(scores_str)
    else:
        raise ValueError("Scores dictionary not found in the prompt")
    
    if profile_match:
        profile_str = profile_match.group(1).strip()
    else:
        raise ValueError("Profile not found in the prompt")

    return scores_dict, profile_str


def extract_intention_approval(text):
    """
    Extract the 'yes/no' responses for intentions and reasons for intentions from the given text.
    """
    lines = text.split('\n')
    intentions = []
    reasons_intention = []
    is_reason_section_intention = False

    for line in lines:
        line = line.strip()
        if line.startswith("Intentions:"):
            intentions = line.replace("Intentions:", "").strip().strip("[]").split(", ")
        elif line.startswith("Reasons_intentions:"):
            is_reason_section_intention = True  # Start extracting reasons for intentions
        elif is_reason_section_intention and line:
            reasons_intention.append(line.strip())

    return intentions, reasons_intention


def extract_predicate_approval(text):
    """
    Extract the 'yes/no' responses for tasks and reasons for tasks from the given text.
    """
    lines = text.split('\n')
    tasks = []
    reasons_tasks = []
    is_reason_section_tasks = False

    for line in lines:
        line = line.strip()
        if line.startswith("Tasks:"):
            tasks = line.replace("Tasks:", "").strip().strip("[]").split(", ")
        elif line.startswith("Reasons_tasks:"):
            is_reason_section_tasks = True  # Start extracting reasons for tasks
        elif is_reason_section_tasks and line:
            reasons_tasks.append(line.strip())

    return tasks, reasons_tasks


def extract_inhand_obj_human(objects_list):
    """
    Extract the inhand_obj_name from each item in the list of object descriptions.
    """
    inhand_obj_names = []
    
    for obj in objects_list:
        # Find the substring that starts with 'inhand_obj_name:' and extract its value
        start_index = obj.find("inhand_obj_name:") + len("inhand_obj_name:")
        end_index = obj.find(",", start_index)
        if end_index == -1:  # Handle the case where inhand_obj_name is at the end
            end_index = obj.find("]", start_index)
        inhand_obj_name = obj[start_index:end_index].strip()
        inhand_obj_names.append(inhand_obj_name)
    
    return inhand_obj_names


def extract_inhand_obj_robot(input_list):
    values = []
    for item in input_list:
        if '[' in item and ']' in item:
            # Remove the outer square brackets
            stripped_item = item.strip('[]').strip()
            
            # Attempt to split by ": "
            if ': ' in stripped_item:
                parts = stripped_item.split(': ', 1)
                
                # If we have at least two parts, use the second
                if len(parts) > 1:
                    values.append(parts[1].strip())
                else:
                    # Fallback: just use everything inside the brackets
                    values.append(stripped_item)
            else:
                # No ": " found, just use everything inside the brackets
                values.append(stripped_item)
        else:
            # No square brackets found, use the item as is
            values.append(item)
            
    return values