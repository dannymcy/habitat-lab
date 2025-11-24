from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os
import time
import json
import torch, gc
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image


# Load environment variables from .env file
load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

def wait_for_finetune_completion_openai(job_id, model_path):
    """
    Poll the fine-tuning job status until it completes, then extract and save the fine-tuned model name.

    Args:
    - job_id (str): The ID of the fine-tuning job.
    - model_path (str): Path to save the fine-tuned model name.
    """
    while True:
        # Retrieve the current status of the fine-tuning job
        finetune_job = client.fine_tuning.jobs.retrieve(job_id)
        job_status = finetune_job.status

        print(f"Job Status: {job_status}")

        if job_status == "succeeded":
            # Extract the fine-tuned model name
            finetuned_model_name = finetune_job.fine_tuned_model
            print(f"Fine-tuned Model Name: {finetuned_model_name}")

            # Save the fine-tuned model name for future use
            with open(model_path, "w") as f:
                f.write(finetuned_model_name)

            return finetuned_model_name
        
        elif job_status == "failed":
            raise Exception(f"Fine-tuning job failed: {finetune_job.error}")
        
        # Wait for some time before polling again
        time.sleep(30)  # Poll every 30 seconds


def finetune_openai(data, suffix, model_path, n_epochs=1, model="gpt-4o-mini-2024-07-18"):
    file_upload = client.files.create(
        file=open(data, "rb"),
        purpose="fine-tune"
    )
    file_id = file_upload.id

    hyperparameters = {
        "n_epochs": n_epochs
    }

    finetune_job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=suffix,
        hyperparameters=hyperparameters,
        seed=42
    )
    finetuned_model_name = wait_for_finetune_completion(finetune_job.id, model_path)

    return finetuned_model_name


def load_finetuned_model_name_openai(model_path):
    # Read the model name from the file
    with open(model_path, "r") as f:
        finetuned_model_name = f.read().strip()
    return finetuned_model_name


def create_data_openai(prompts, responses, json_path=None, training=True):
    """
    Create structured data for prompt-response pairs for fine-tuning or inference.

    Args:
    - prompts (list of str): A list of user prompts.
    - responses (list of str): A list of assistant responses.
    - json_path (str): Path to save the generated JSONL file if in training mode.
    - training (bool): If True, save the data as JSONL. If False, return structured data for inference.

    Returns:
    - If training is False, returns the structured data without saving to a file.
    """
    # Ensure that the prompts and responses lists are of equal length
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses must have the same length.")
    
    data = []
    
    # Iterate through the prompts and responses to build the structured data
    for prompt, response in zip(prompts, responses):
        conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        }
        data.append(conversation)
    
    # If training, save the data to a JSONL file
    if training:
        with open(json_path, 'w') as f:
            for conversation in data:
                f.write(json.dumps(conversation) + '\n')  # Write each dict as a JSON line
        print(f"Data has been written to {json_path}")
    else:
        # If not training, return the structured data for inference
        return data


def query(system, user_contents, assistant_contents, save_path=None, model='gpt-4', temperature=1, debug=False, max_retries=20):
    for user_content, assistant_content in zip(user_contents, assistant_contents):
        user_content = user_content[0].split("\n")
        assistant_content = assistant_content[0].split("\n")
        
        for u in user_content:
            print(u)
        print("=====================================")
        for a in assistant_content:
            print(a)
        print("=====================================")

    for u in user_contents[-1][0].split("\n"):
        print(u)

    if debug:
        import pdb; pdb.set_trace()
        return None

    print("=====================================")

    start = time.time()
    
    num_assistant_mes = len(assistant_contents)
    messages = []

    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx][0]})
        if user_contents[idx][1]:
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}]
            for image_url in user_contents[idx][1]:
                messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})
            
        messages.append({"role": "assistant", "content": assistant_contents[idx][0]})
        if assistant_contents[idx][1]:
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}]
            for image_url in assistant_contents[idx][1]:
                messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})
    messages.append({"role": "user", "content": user_contents[-1][0]})
    
    # Add the base64 encoded image to the last user message
    if user_contents[-1][1]:
        messages[-1]["content"] = [
            {"type": "text", "text": messages[-1]["content"]}]
        for image_url in user_contents[-1][1]:
            messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})

    retries = 0

    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=4096
            )

            result = ''
            for choice in response.choices: 
                result += choice.message.content 

            end = time.time()
            used_time = end - start

            print(result)

            user_contents_text, assistant_contents_text = [], []
            for user_content in user_contents:
                user_contents_text.append(user_content[0])
            for assistant_content in assistant_contents:
                assistant_contents_text.append(assistant_content[0])

            if save_path is not None:
                with open(save_path, "w") as f:
                    json.dump({"used_time": used_time, "res": result, "system": system, "user": user_contents_text, "assistant": assistant_contents_text}, f, indent=4)
                with open(save_path, 'r') as f:
                    json_data = json.load(f)
            
            return json_data

        except OpenAIError as e:
            print(f"Error occurred: {str(e)}. Retrying...")
            retries += 1
            sleep(2)  # Adding a delay before retrying

    print(f"Failed after {max_retries} attempts.")
    return None


def clear_gpu_memory(model, trainer, tokenizer):
    # Step 1: Move any remaining models/tensors to CPU
    torch.cuda.empty_cache()

    # Step 2: Force garbage collection to clear up memory
    gc.collect()

    # Step 3: If using a model, explicitly move to CPU and delete
    # For example, if your model is still in memory:
    model.cpu()
    del model
    if trainer is not None: del trainer
    del tokenizer

    # Step 4: Clear the GPU cache
    torch.cuda.empty_cache()

    # Step 5: (Optional) Reset device states
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()


def llm_local_inference(user_content, image_paths=None, save_path=None, temperature=0.2, max_tokens=4096):
    """
    Run local Llama inference with optional image support
    
    Args:
        user_content (str): The text prompt/content
        image_paths (list, optional): List of image paths for multimodal input
        save_path (str, optional): Path to save the response JSON
        temperature (float): Generation temperature (default: 0.2)
        max_tokens (int): Maximum tokens to generate (default: 4096)
    
    Returns:
        dict: Response data with timing, response text, system prompt, and user content
    """
    ts = time.time()
    
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=os.environ['HUGGINGFACE_TOKEN'],  
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        device_map='balanced'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=os.environ['HUGGINGFACE_TOKEN'],
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        use_auth_token=os.environ['HUGGINGFACE_TOKEN']
    )
    
    # Prepare image if provided
    image = None
    if image_paths and len(image_paths) > 0:
        # Use the second image if available (index 1), otherwise first
        img_index = 1 if len(image_paths) > 1 else 0
        image = Image.open(image_paths[img_index]).convert("RGB")
    
    # Construct message for Llama-3 Vision
    messages = [{"role": "user", "content": []}]
    
    if image:
        messages[0]["content"].append({"type": "image", "image": image})
    
    messages[0]["content"].append({"type": "text", "text": user_content})
    
    # Prepare input prompt
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=temperature
        )
    
    # Decode response
    prompt_ids = inputs["input_ids"][0]
    full_output_ids = outputs[0]
    generated_ids = full_output_ids[len(prompt_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Prepare response data
    json_data = {
        "used_time": time.time() - ts, 
        "res": generated_text, 
        "system": "You are a helpful assistant.", 
        "user": user_content
    }
    
    # Save if path provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(json_data, f, indent=4)
    
    print(generated_text)
    print()
    
    # Clean up GPU memory
    clear_gpu_memory(model, processor, tokenizer)
    
    return json_data