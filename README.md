# RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models
Authors: 

Yeongtak Oh, Jisoo Mok, Dohyun Chung, Juhyeon Shin, Sangha Park, Johan Barthelemy, and Sungroh Yoon

Affiliation: 

Seoul National University

Abstract: 

Recent multi-modal large language models (MLLMs) often struggle to generate personalized image captions, even when trained on high-quality captions. In this work, we observe that such limitations persist in existing post-training-based MLLM personalization methods. Specifically, despite being post-tuned with large-scale caption data through supervised fine-tuning (SFT), these models frequently fail to produce faithful descriptions in real-world scenarios, such as multi-concept image captioning. However, acquiring large-scale, high-quality captions for such complex settings is both costly and difficult. To address the data-centric nature of SFT, we propose a reinforcement learning (RL)-based post-training framework. To the best of our knowledge, this is the first RL-based approach to post-train MLLMs for personalized image captioning. Our method significantly enhances both visual recognition and personalized generation capabilities of MLLMs, and consistently outperforms existing SFT-based baselines, especially in the challenging multi-concept image captioning task. 

<p align="center">
  <img src="./__assets__/teaser_renew_1-1.png" alt="Figure 1. Visualizations of personalized image captioning results. In the first row, the zero-shot MLLM frequently fails to generate a personalized caption. The remaining rows illustrate multi-concept scenarios at inference time. Compared to other SFT-based methods, our approach consistently produces faithful and detailed captions while accurately recognizing all provided identities, even for 3 or 4 concepts" width="80%">
</p>

## 📦 Installation Guide (Inference Only)

Our codebase has been tested on **CUDA 12.4**. Please follow the instructions below:

```bash
# Create and activate conda environment
conda create -n RePIC python=3.11 -y
conda activate RePIC

# Install CUDA 12.4 toolkit
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

# Setup permissions and dependencies
chmod 755 *.sh
bash ./setup.sh

# Set up kernel
conda install ipykernel -y
python -m ipykernel install --user --name RePIC --display-name RePIC
```

> ⚠️ If installation fails, it may be due to issues with the `flash_attention_2` library.  
Please refer to the official [Qwen2.5-VL repository](https://github.com/QwenLM/Qwen2.5-VL) for alternative inference guidance.

We have only tested inference with the **`flash_attention_2`** setup. Logs and example outputs are included in [`inference_example.ipynb`](./inference_example.ipynb).

---

## 🖼️ Visualization Example

You can run the `gradio_example_4_concept.ipynb` notebooks for **visualization with pre-generated captions** without installing the environmental settings.

> 📌 Note: We curated the database and query images for a 4-concept setting, all evaluation images are credited to [RAP-MLLM](https://arxiv.org/abs/2410.13360).

Feel free to try it out!

---

## 🧪 Inference Example

The `inference_example.ipynb` notebook contains:
- Scripts to run **inference with your own queries**
- Reproducible code for **Figure 1, Figure A.1 and Figure A.2** in our paper

---

## 🏋️ Training & Evaluation

Training and evaluation code is currently **under construction**.  
We plan to release it as open source in the near future!

---

## 📁 Data Download

You can download our **5K dataset** used for training and evaluation here:

📎 [Google Drive Link](https://drive.google.com/file/d/1DKPmLI58NZUpSUFEUgzpUmMKKPK3Oguc/view?usp=sharing)

> ✅ Note: We only used a **2K subset** of this dataset for training purposes.

---

## 🙏 Acknowledgements

We gratefully acknowledge the following open-source repositories and resources that supported our work:

- 🔗 [VLM-R1 (Om-AI-Lab)](https://github.com/om-ai-lab/VLM-R1)  
- 🔗 [Qwen2.5-VL (QwenLM)](https://github.com/QwenLM/Qwen2.5-VL)
