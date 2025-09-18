export CUDA_VISIBLE_DEVICES=0

# 1) 2-concept personalization
# Skip-Retrieval
python caption_eval_multi_2.py \
  --modelname RePIC \
  --datasetname 2_concepts \
  --topK 2

# Retrieval
python caption_eval_multi_2.py \
  --retrieval_mode \
  --modelname RePIC \
  --datasetname 2_concepts \
  --topK 2

# 1) 4-concept personalization
# Skip-Retrieval
python caption_eval_multi_4.py \
  --modelname RePIC \
  --datasetname 4_concepts \
  --topK 4

# Retrieval
python caption_eval_multi_4.py \
  --retrieval_mode \
  --modelname RePIC \
  --datasetname 4_concepts \
  --topK 4