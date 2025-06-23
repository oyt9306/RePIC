# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the packages in open-r1-multimodal .
pip install -e ".[dev]"

# Addtional modules
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
