# eval_captioning_multi_concept.py
# Copyright: Yeongtak Oh
# Github: https://github.com/oyt9306/RePIC

import argparse
import torch
import requests
import json
import os
import random
import copy
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from detector import Detector
from retriever import ClipRetriever
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def sample_from_dict(d, sample=1):
    """Randomly sample key-value pairs from a dictionary."""
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return keys[0], values[0]


def load_image(image_file):
    """Load an image from a local path or URL."""
    if image_file.startswith(('http://', 'https://')):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def run_captioning(modelname, dataset_name, retrieval_mode, topK, index_path, device):


    # Caption prompt candidates
    caption_prompts = [
        "Give a personalized caption of this image.",
        "Give a personal caption of the image.",
        "Provide a personalized caption of the given image.",
    ]

    # Model selection
    if modelname == 'RePIC':
        MODEL_PATH = 'path/to/LoRA/ckpt'
        QUESTION_TEMPLATE = "{Question}"
    elif modelname == 'RePIC_huggingface':
        MODEL_PATH = 'Yeongtak/RePIC_Qwen2.5VL_7B'
        QUESTION_TEMPLATE = "{Question}"
    elif modelname == 'zero_shot':
        MODEL_PATH = 'Qwen/Qwen2.5-VL-7B-Instruct'
        QUESTION_TEMPLATE = "{Question} Output the final answer including its name."
    else:
        raise ValueError(f"Unsupported model name: {modelname}")

    # Target dataset
    database_path = './data/database_tot/database_4_concepts'            
    eval_file = './data/query_tot/4_concept_query_templates.json'
    
    # Load evaluation set
    with open(eval_file, "r") as f:
        test_set = json.load(f)
    
    # Load model & processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, max_pixels=1024 * 28 * 28)

    # Load database & retriever
    with open(f"{database_path}/database_mod.json", "r") as f:
        database = json.load(f)

    retriever = ClipRetriever(
        data_dir=database_path,
        index_path=index_path,
        create_index=index_path is None
    )

    # Optional: open-vocabulary detection
    if retrieval_mode:
        detector = Detector()
        all_category = list({database["concept_dict"][c]["category"] for c in database["concept_dict"]})
        detector.model.set_classes(all_category)

    # Caption generation loop
    sys_prompt = (
        "You are a captioning assistant. Your task is to generate an accurate caption "
        "for the query image.\nBelow is additional information about the reference images."
    )
    results = []
    cnt = 0
    empty_image = {"type": "image", "image": ""}
    empty_text = {"type": "text", "text": ""}

    for img_path, concepts_tot in tqdm(test_set.items()):
        for concepts in concepts_tot:

            message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": []}
            ]

            image = load_image(img_path)

            # Retrieval step
            if retrieval_mode:
                crops = detector.detect_and_crop(image)
                extra_info, rag_images = retriever.retrieve(database, inp="", queries=crops, topK=topK)
            else:
                extra_info, rag_images = retriever.retrieve(database, inp=concepts, queries=[], topK=topK)

            extra_info = extra_info.replace("<image>\n", "")

            # Append retrieved images & info
            for i, ret_path in enumerate(rag_images):
                temp_img = copy.deepcopy(empty_image)
                temp_img['image'] = ret_path
                message[1]['content'].append(temp_img)

                temp_txt = copy.deepcopy(empty_text)
                temp_txt['text'] = extra_info.split('\n')[i]
                message[1]['content'].append(temp_txt)

            # Append query image
            query_img = copy.deepcopy(empty_image)
            query_img['image'] = img_path
            message[1]['content'].append(query_img)

            question = caption_prompts[cnt % len(caption_prompts)]
            cnt += 1

            query_txt = copy.deepcopy(empty_text)
            query_txt['text'] = f"This is the query image. {QUESTION_TEMPLATE.format(Question=question)}"
            message[1]['content'].append(query_txt)

            # Process inputs
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(message)

            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate caption
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(outputs)
            results.append([img_path, concepts, outputs, question])

        save_path = f"./save_script/multi_captioning-retrieval-{retrieval_mode}-{modelname}-{dataset_name}.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-concept Image Captioning Evaluation")
    parser.add_argument("--modelname", type=str, default="RePIC", choices=["RePIC", "RePIC_huggingface", "RAP_Qwen", "zero_shot"])
    parser.add_argument("--datasetname", type=str, default="2_concepts", choices=["2_concepts", "4_concepts"])
    parser.add_argument("--retrieval_mode", action="store_true", help="Enable retrieval mode")
    parser.add_argument("--topK", type=int, default=2, help="Number of concepts to retrieve")
    parser.add_argument("--index_path", type=str, default=None, help="Path to CLIP index file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for model execution")
    args = parser.parse_args()

    if not os.path.exists('./save_script'):
        os.mkdir('./save_script')
        
    run_captioning(
        modelname=args.modelname,
        dataset_name=args.datasetname,
        retrieval_mode=args.retrieval_mode,
        topK=args.topK,
        index_path=args.index_path,
        device=args.device
    )


if __name__ == "__main__":
    main()