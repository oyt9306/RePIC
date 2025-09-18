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
    
with open('./data/database_tot/database_2_concepts/database_mod.json', "r") as f:
    database = json.load(f)

outputs = []

for method in ['zero_shot', 'rap_llava', 'rap_qwen_sft']:
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

        # db 1
        image_path = database['concept_dict'][ours[idx][1][0]]['image']
        info1      = database['concept_dict'][ours[idx][1][0]]['info']
        with open(image_path, 'rb') as f:
            base64_db1 = f.read()
    
        # db 2
        image_path = database['concept_dict'][ours[idx][1][1]]['image']
        info2      = database['concept_dict'][ours[idx][1][1]]['info']
        with open(image_path, 'rb') as f:
            base64_db2 = f.read()
            
        chat_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                    f"You are an evaluation expert. Your task is to determine which answer best describes the given query image accurately. Carefully analyze the options and select the most appropriate one as your final choice. \n \
                    The preferable caption for the query image is one that is not merely a duplication of the given information but provides a meaningful and accurate description of the image. \n \
                    The first image is the query image. \n \
                    The second image is the image of {ours[idx][1][0]} with the given Info: {info1}. \n \
                    The third image is the image of {ours[idx][1][1]} with the given Info: {info2}. \n \
                    Options: \n A: {string1} \n B: {string2} \n Referencing the second and the third images, output the final answer by choosing one of the options with a single alphabet.",
                    base64_qwen,
                    types.Part.from_bytes(
                        data=base64_db2,
                        mime_type='image/png'
                    ),
                    types.Part.from_bytes(
                        data=base64_db2,
                        mime_type='image/png'
                    )
                ]
            )
        response = chat_response.text
        print("Chat response:", response)
        outputs.append(response)
        
    with open(f"./results/gemini_multi_preference_{method}_ours_qwenVL_7B.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)