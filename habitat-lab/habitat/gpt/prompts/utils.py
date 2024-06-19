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
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
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


# def extract_useful_object(prompt_name, prompt_path):
#     if prompt_path.exists():
#         subdirs = [d for d in os.listdir(prompt_path) if os.path.isdir(prompt_path / d)]
#         subdirs.sort()

#     # Find the latest subdirectory
#     latest_subdir = max(subdirs, key=lambda d: (prompt_path / d).stat().st_mtime)
#     json_file_path = prompt_path / latest_subdir / f"{prompt_name}.json"

#     if json_file_path.exists():
#         with open(json_file_path, 'r') as f:
#             json_data = json.load(f)
#             res_text = json_data["res"]

#             # Extract object descriptions using regular expression
#             object_descriptions = re.findall(r'Object \d+: (.+?)(?=\n|$)', res_text)
#             return object_descriptions


# def extract_words_before(sentence, cutoff_word):
#     words = sentence.split()
#     cutoff_index = words.index(cutoff_word)
#     return ' '.join(words[:cutoff_index])


def parse_planning_line(line):
    line = line.strip().lstrip('[').rstrip(']')
    elements = line.split("), (")
    parsed_elements = []

    for element in elements:
        element = element.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        parts = element.split(", ")
        parts = [p.strip().strip("'").strip() for p in parts]

        # Determine if parts contain an object name with commas
        if len(parts) > 4:
            parts = [parts[0]] + [", ".join(parts[1:-2])] + parts[-2:]
        
        if len(parts) == 4:
            parts[0] = parts[0].split(". ")[-1]  # Remove the index if present
            parsed_elements.append((parts[0], parts[1], int(parts[2]), parts[3]))
        else:
            parts[0] = parts[0].split(". ")[-1]  # Remove the index if present
            parsed_elements.append(tuple(parts))

    return parsed_elements

def extract_code(prompt_name, prompt_path, video_path=None, scene_id=None):
    # Create the video path directory if it doesn't exist
    # video_path.mkdir(parents=True, exist_ok=True)

    if prompt_path.exists():
        json_file_path = load_response(prompt_name, prompt_path)
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Initialize the dictionary
        result_dict = {}

        # Extract the "res" field content
        res_content = json_data["res"]
        
        # Split the content by time sections
        time_sections = res_content.split("\n\nTime: ")
        
        for section in time_sections:
            if section.strip() == "":
                continue

            lines = section.split("\n")
            time = lines[0].strip()  # Ensure 'Time: ' prefix is included
            intention = lines[1].replace("Intention: ", "").strip()
            
            predicates = []
            planning = []
            predicate_section = False
            planning_section = False

            for line in lines[2:]:
                if line.startswith("Predicates:"):
                    predicate_section = True
                    planning_section = False
                    continue
                elif line.startswith("Planning:") or line.startswith("Planning (corresponding to each predicate):"):
                    planning_section = True
                    predicate_section = False
                    continue
                
                if predicate_section:
                    predicate = line.strip()
                    if predicate[0].isdigit() and predicate[1] == '.':
                        predicate = predicate.split('. ', 1)[-1].strip()  # Remove index
                    predicates.append(predicate)
                elif planning_section:
                    plan = parse_planning_line(line.strip())
                    cleaned_plan = []
                    for tpl in plan:
                        if len(tpl) == 4:
                            cleaned_plan.append((tpl[0], tpl[1], int(tpl[2]), tpl[3]))
                        else:
                            cleaned_plan.append((tpl[0],) + tpl[1:])
                    planning.append(cleaned_plan)
            
            result_dict[time] = {
                "Intention": intention,
                "Predicates": predicates,
                "Planning": planning
            }

        return result_dict