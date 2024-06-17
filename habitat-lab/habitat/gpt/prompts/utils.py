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


def extract_useful_object(prompt_name, prompt_path):
    if prompt_path.exists():
        subdirs = [d for d in os.listdir(prompt_path) if os.path.isdir(prompt_path / d)]
        subdirs.sort()

    # Find the latest subdirectory
    latest_subdir = max(subdirs, key=lambda d: (prompt_path / d).stat().st_mtime)
    json_file_path = prompt_path / latest_subdir / f"{prompt_name}.json"

    if json_file_path.exists():
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            res_text = json_data["res"]

            # Extract object descriptions using regular expression
            object_descriptions = re.findall(r'Object \d+: (.+?)(?=\n|$)', res_text)
            return object_descriptions


def extract_words_before(sentence, cutoff_word):
    words = sentence.split()
    cutoff_index = words.index(cutoff_word)
    return ' '.join(words[:cutoff_index])


def extract_code(prompt_name, prompt_path, video_path, img_id):
    video_path.mkdir(parents=True, exist_ok=True)
    if prompt_path.exists():
        subdirs = [d for d in os.listdir(prompt_path) if os.path.isdir(prompt_path / d)]
        subdirs.sort()

        generated_codes = []
        generated_tasks = []
        video_filenames = []

        for i, subdir in enumerate(subdirs):
            json_file_path = prompt_path / subdir / f"{prompt_name}.json"
            if json_file_path.exists():
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                res_text = json_data["res"]

                # Regular expression patterns
                obj_idx_pattern = r"obj_idx = (\d+)"
                motion_list_pattern = r"motion_list = \[((?:\s*\([^)]*\),?)+)\]"
                task_pattern = r"Task Name: (.*?)\nDescription: (.*?)\n"

                # Finding all matches for codes
                obj_idx_matches = re.findall(obj_idx_pattern, res_text)
                motion_list_matches = re.findall(motion_list_pattern, res_text, re.DOTALL)

                # Finding all matches for tasks
                task_matches = re.findall(task_pattern, res_text, re.DOTALL)

                # Process and store code data
                for obj_idx, motion_list_str in zip(obj_idx_matches, motion_list_matches):
                    # Convert motion_list string to tuple
                    motion_list = ast.literal_eval(f'[{motion_list_str}]')
                    generated_codes.append((int(obj_idx), motion_list))

                # Process and store task data
                generated_tasks.extend(task_matches)

            # Write to a .txt file
            if task_matches:  # Check if there are tasks to write
                for task in task_matches:
                    txt_filename = f"task_{img_id}_obj_{i}_{task[0].replace(' ', '_')}.txt"
                    video_filename = f"video_{img_id}_obj_{i}_{task[0].replace(' ', '_')}.mp4"
                    video_filenames.append(video_filename)
                    with open(video_path / txt_filename, 'w') as file:
                        file.write(f"Task Name: {task[0]}\nDescription: {task[1]}\n")

        return generated_codes, video_filenames