# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original Source Code : https://github.com/om-ai-lab/VLM-R1/blob/main/src/open-r1-multimodal/src/open_r1/grpo_rec.py

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import random

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for data in datasets:
                    json_path = data.get("json_path")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    # random.shuffle(cur_data_dict)
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
                random.shuffle(self.list_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Build a multimodal training/eval sample:
          - Loads one or more images
          - Chooses a prompt template based on `accu_reward_method`
          - Optionally randomizes the instruction to reduce overfitting
          - Returns image(s), problem/solution, prompt, and method
        """
        example = self.list_data_dict[i]
        assert "image" in example, "Missing 'image' in example."

        image_root = self.script_args.image_root

        # Load image(s)
        if isinstance(example["image"], list):
            image_paths: List[str] = [os.path.join(image_root, p) for p in example["image"]]
            images: List[Image.Image] = [Image.open(p).convert("RGB") for p in image_paths]
            num_images = len(images)
        else:
            image_paths = os.path.join(image_root, example["image"])
            images = Image.open(image_paths).convert("RGB")
            num_images = 1

        # Default method
        method = example.get("accu_reward_method", "iou")
        example["accu_reward_method"] = method

        question = example["problem"]

        # ---------------------------
        # Prompt builders
        # ---------------------------
        def _randomized_free_text(q: str) -> str:
            """Randomize to mitigate overfitting."""
            roll = random.randint(0, 4)
            if roll == 0:
                template = "{q} Output the final answer including its name."
            elif roll == 1:
                template = "{q} Output the final answer without duplicating the given information."
            else:
                # Important: Refer with the name without explicitly applying naming prompt
                template = "{q}"
            return template.format(q=q)

        def _text_iou(q: str) -> str:
            return "{q} For bounding box coordinates, return the final answer in JSON format.".format(q=q)

        def _text_yes_no(q: str) -> str:
            return "{q} The final answer must be either 'yes' or 'no' with no additional output.".format(q=q)

        def build_prompt(n_images: int, text: str) -> Dict[str, Any]:
            contents = [{"type": "image"} for _ in range(n_images)]
            contents.append({"type": "text", "text": text})
            return {"prompt": [{"role": "user", "content": contents}]}

        # ---------------------------
        # Route by method & #images
        # ---------------------------
        if num_images > 1:
            # Multi-image inputs
            if method in ("yes_no_name_multi", "yes_no_name"):
                prompt = build_prompt(num_images, _randomized_free_text(question))
            elif method == "yes_no":
                prompt = build_prompt(num_images, _text_yes_no(question))
            else:
                # Fallback: historical behavior defaulted to IOU builder (single image token)
                prompt = build_prompt(1, _text_iou(question))
        else:
            # Single-image input
            if method == "yes_no":
                prompt = build_prompt(1, _text_yes_no(question))
            elif method == "iou":
                prompt = build_prompt(1, _text_iou(question))
            elif method == "yes_no_name":
                prompt = build_prompt(1, _randomized_free_text(question))
            else:
                # Sensible fallback
                prompt = build_prompt(1, _text_iou(question))

        return {
            "image": images,
            "problem": example["problem"],
            "solution": example["solution"],
            "prompt": prompt["prompt"],
            "accu_reward_method": example["accu_reward_method"],
        }

# -----------------------------
# Constants (tuning parameters)
# -----------------------------
MIN_CONTENT_LEN = 100          # Minimum output length to consider reward valid
IOU_THRESHOLD   = 0.5            # IoU threshold for correct answer
BBOX_REGEX      = r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]"  # Regex pattern to match [x1, y1, x2, y2]

# -----------------------------
# 1) Verifiable reward for OCT
# -----------------------------
@torch.no_grad()
def yes_no_reward(content: str, sol: str, **kwargs) -> float:
    """
    OCT case: Checks if both ground truth and predicted output are 'yes' or 'no' and match.
    - Ignores case sensitivity.
    - Extracts 'yes' or 'no' if present anywhere in the string.
    - Returns 1.0 if they match, otherwise 0.0.
    """
    gt = (sol or "").lower()
    out = (content or "").lower()

    gt_m = re.search(r"(yes|no)", gt, flags=re.IGNORECASE)
    out_m = re.search(r"(yes|no)", out, flags=re.IGNORECASE)

    gt_ans = (gt_m.group(1).lower() if gt_m else "")
    out_ans = (out_m.group(1).lower() if out_m else "")

    return 1.0 if gt_ans and out_ans and (gt_ans == out_ans) else 0.0


# --------------------------------
# 2) Verifiable reward for single-ICT
# --------------------------------
@torch.no_grad()
def yes_no_answer_reward(content: str, sol: str, **kwargs) -> float:
    """
    ICT (single-answer) case: Checks if the exact ground truth string exists in the model output.
    - Applies length regularization: output must be at least MIN_CONTENT_LEN characters.
    - Prevents reward hacking: if the ground truth starts with '<', the closing '</' tag 
      should not also be included in the prediction.
    - Uses exact string matching with re.escape to avoid regex special character issues.
    """
    if not isinstance(content, str) or not isinstance(sol, str):
        return 0.0

    if len(content) < MIN_CONTENT_LEN:
        return 0.0

    found = re.search(re.escape(sol), content) is not None

    if sol.startswith("<"):  # Prevents reward hacking for tags like <name>
        hacked = re.search(re.escape(sol.replace("<", "</")), content) is not None
        return 1.0 if (found and not hacked) else 0.0

    return 1.0 if found else 0.0


# --------------------------------
# 3) Verifiable reward for multi-ICT
# --------------------------------
@torch.no_grad()
def yes_no_answer_reward_multi(content: str, sol: Sequence[str], **kwargs) -> float:
    """
    ICT (multi-answer) case: Checks if each answer token in 'sol' exists in the output.
    - Returns the average success rate across all tokens.
    - Applies length regularization: output must be at least MIN_CONTENT_LEN characters.
    - Uses exact string matching with re.escape.
    """
    if not isinstance(content, str) or not isinstance(sol, (list, tuple)):
        return 0.0

    if len(content) < MIN_CONTENT_LEN:
        return 0.0

    if len(sol) == 0:
        return 0.0

    hits = [(re.search(re.escape(s), content) is not None) for s in sol]
    return float(sum(1 for h in hits if h)) / float(len(hits))


# -----------------------------
# 4) Verifiable reward for IOU
# -----------------------------
def _iou(box1: Sequence[int], box2: Sequence[int]) -> float:
    """
    Calculates IoU between two bounding boxes.
    - Boxes are in the format [x1, y1, x2, y2] with inclusive coordinates.
    - Returns IoU value between 0.0 and 1.0.
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12 - 1, x22 - 1)
    inter_y2 = min(y12 - 1, y22 - 1)

    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0

    area1 = max(0, (x12 - x11)) * max(0, (y12 - y11))
    area2 = max(0, (x22 - x21)) * max(0, (y22 - y21))
    union = max(1, area1 + area2 - inter)  # Avoid division by zero

    return float(inter) / float(union)


@torch.no_grad()
def iou_reward(content: str, sol: Sequence[int], **kwargs) -> float:
    """
    IoU case: Extracts the predicted bounding box from model output and compares it to the ground truth.
    - The predicted box is extracted using a regex pattern.
    - Returns 1.0 if IoU is above IOU_THRESHOLD, else 0.0.
    """
    if not isinstance(content, str) or not (isinstance(sol, (list, tuple)) and len(sol) == 4):
        return 0.0

    m = re.search(BBOX_REGEX, content)
    if not m:
        return 0.0

    pred = [int(m.group(i)) for i in range(1, 5)]
    return 1.0 if _iou(pred, sol) >= IOU_THRESHOLD else 0.0


@torch.no_grad()
def accuracy_reward(
    completions: Sequence[Sequence[Dict[str, str]]],
    solution: Sequence[Union[str, Sequence[str], Sequence[int]]],
    **kwargs: Any
) -> List[float]:
    """
    Calculates rewards for a batch of predictions using the appropriate reward function.
    
    Parameters:
        completions: List of samples, each containing a list of dicts with 'content' as the model output.
        solution: List of ground truth answers (string, list of strings, or bounding box coordinates).
        kwargs["accu_reward_method"]: List of methods for each sample. Supported:
            - "yes_no"
            - "yes_no_name"
            - "yes_no_name_multi"
            - "iou"
    
    Returns:
        rewards: List of reward values for each sample.
    
    Debugging:
        If DEBUG_MODE=true in environment variables, logs detailed reward calculation info to LOG_PATH.
    """
    methods: Sequence[str] = kwargs.get("accu_reward_method", [])
    contents = [c[0]["content"] if c and isinstance(c[0], dict) else "" for c in completions]

    rewards: List[float] = []
    for idx, (content, sol, method) in enumerate(zip(contents, solution, methods)):
        if method == "yes_no":
            r = yes_no_reward(content, str(sol))
        elif method == "yes_no_name_multi":
            r = yes_no_answer_reward_multi(content, sol if isinstance(sol, (list, tuple)) else [str(sol)])
        elif method == "iou":
            r = iou_reward(content, sol if isinstance(sol, (list, tuple)) else [])
        elif method == "yes_no_name":
            r = yes_no_answer_reward(content, str(sol))
        else:
            r = 0.0

        rewards.append(float(r))

        # Optional logging
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                image_path = (kwargs.get("image_path") or [None])[0]
                problem = (kwargs.get("problem") or [None])[0]
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {r} -------------\n")
                    f.write(f"accu_reward_method: {method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
