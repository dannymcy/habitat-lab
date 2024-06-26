from openai import OpenAI
import os
import time
import json

# key name: danny_mani_vid
os.environ["OPENAI_API_KEY"] = ""  # put your api key here
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

def query(system, user_contents, assistant_contents, save_path=None, model='gpt-4', temperature=1, debug=False):
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
                {"type": "text", "text": messages[-1]["content"]},
                {"type": "image_url", "image_url": user_contents[idx][1]}
            ]
        messages.append({"role": "assistant", "content": assistant_contents[idx][0]})
        if assistant_contents[idx][1]:
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]},
                {"type": "image_url", "image_url": assistant_contents[idx][1]}
            ]
    messages.append({"role": "user", "content": user_contents[-1][0]})
    
    # Add the base64 encoded image to the last user message
    if user_contents[-1][1]:
        messages[-1]["content"] = [
            {"type": "text", "text": messages[-1]["content"]},
            {"type": "image_url", "image_url": user_contents[-1][1]}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=4096
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