from openai import OpenAI
import json
import base64
import re
from tqdm import tqdm 
import os

if not os.path.exists('./results'):
    os.mkdir('./results')

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = ""
client = OpenAI(
    api_key=openai_api_key,
)

dataname = 'myvlm'

with open('path/to/your/generated/captions.json', "r") as f:
    ours = json.load(f)

with open('path/to/your/generated/captions.json', "r") as f:
    zero_shot = json.load(f)
    


outputs = []

for method in ['zero_shot']:
    for idx in tqdm(range(len(ours))):

        string1 = f'Ans: \n{zero_shot[idx][2]}'
        string2 = f'Ans: {ours[idx][2]}'
        
        image_path = ours[idx][0]
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image/png;base64,{encoded_image_text}"
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are an evaluation expert. Your task is to determine which answer best describes the given image accurately. Carefully analyze the options and select the most appropriate one as your final choice."},            {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"\nThe preferable caption is one that provides a meaningful and accurate description of the image."},
                        {"type": "text", "text": f"Which one is more preferable caption to the image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_qwen
                            },
                        },
                    {"type": "text", "text": f"Options: \n A: {string1} \n B: {string2} \nOutput the final answer by choosing one of the options with a single alphabet."},
                    ],
                },
            ],
        )
        response = chat_response.choices[0].message.content
        print("Chat response:", response)
        outputs.append(response)
        
    with open(f"./results/chatgpt_preference_{dataname}_ours_qwenVL_7B.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)