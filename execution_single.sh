export CUDA_VISIBLE_DEVICES=1

# 1) DreamBooth dataset
# Skip-retrieval
python caption_eval_single.py \
  --datasetname dreambooth \
  --modelname RePIC \
  --topK 2

# Retrieval
python caption_eval_single.py \
  --datasetname dreambooth \
  --retrieval_mode \
  --modelname RePIC \
  --topK 2

# 2) MyVLM dataset
# Skip-retrieval
python caption_eval_single.py \
  --datasetname myvlm \
  --modelname RePIC \
  --topK 2

# Retrieval
python caption_eval_single.py \
  --datasetname myvlm \
  --retrieval_mode \
  --modelname RePIC \
  --topK 2

# 3) Yollava dataset
# Skip-retrieval
python caption_eval_single.py \
  --datasetname yollava \
  --modelname RePIC \
  --topK 2

# Retrieval
python caption_eval_single.py \
  --datasetname yollava \
  --retrieval_mode \
  --modelname RePIC \
  --topK 2
