from google import genai
from google.genai import types
import json 
from tqdm import tqdm 
import os

if not os.path.exists('./results'):
    os.mkdir('./results')
    
client = genai.Client()

with open('path/to/your/generated/captions.json', "r") as f:
    rap_llava = json.load(f)

with open('path/to/your/generated/captions.json', "r") as f:
    ours = json.load(f)

with open('path/to/your/generated/captions.json', "r") as f:
    rap_qwen_sft = json.load(f)

with open('path/to/your/generated/captions.json', "r") as f:
    zero_shot = json.load(f)
    
with open('/path/to/your/database.json', "r") as f:
    database = json.load(f)

outputs = []

for method in ['rap_qwen_sft', 'zero_shot', 'rap_llava']: 
    for idx in tqdm(range(len(ours))):
        if method == 'rap_llava':
           string1 = f'Ans: \n{rap_llava[idx][2]}'
        elif method == 'zero_shot':
           string1 = f'Ans: \n{zero_shot[idx][2]}'
        elif method == 'rap_qwen_sft':
           string1 = f'Ans: \n{rap_qwen_sft[idx][2]}'
        
        string2 = f'Ans: {ours[idx][2]}'
        
        # query 
        image_path = ours[idx][0]
        base64_qwen = client.files.upload(file=image_path)

        chat_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                    f"You are an evaluation expert. Your task is to determine which answer best describes the given image accurately. \
                    Carefully analyze the options and select the most appropriate one as your final choice. \n \
                    The preferable caption is one that provides a meaningful and accurate description of the image. \n \
                    Options: \n A: {string1} \n B: {string2} \n \
                    Output the final answer by choosing one of the options with a single alphabet.",
                    base64_qwen,
                ]
            )
        response = chat_response.text
        print("Chat response:", response)
        outputs.append(response)
        
    with open(f"./results/gemini_single_preference_{method}_ours_qwenVL_7B.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)