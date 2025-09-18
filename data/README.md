# 💾 Dataset Configuration

This document outlines the configuration for the single-concept and multi-concept datasets used in our experiments.

---

## ☝️ Single-Concept Datasets

To run our experiments, please prepare the MyVLM, DreamBooth, and YoLLaVA datasets using the following folder names and download links.

* **MyVLM**
    * **Folder Name**: `myvlm_data`
    * **Download**: 🔗 [Google Drive Link](https://drive.google.com/drive/folders/1dxjwYVAmBRWLeqUjWsR8cWdqMvfsqW79)

* **DreamBooth**
    * **Folder Name**: `dreambooth`
    * **Download**: 🔗 [GitHub Link](https://github.com/google/dreambooth/tree/main/dataset)

* **YoLLaVA**
    * **Folder Name**: `yollava-data`
    * **Download**: 🔗 [Hugging Face Link](https://huggingface.co/datasets/thaoshibe/YoLLaVA)

---

## ✌️ Multi-Concept Datasets


### Data Sources
* **2-Concept Dataset**: First, prepare the 2-concept datasets within the following folders. This dataset was provided directly by the authors of RAP-LLaVA (CVPR 2025). If you use this data in your work, please cite their paper. Download the data from the link below and unzip it into a folder named `query-image-2-concepts`.
    * **Download**: 🔗 [RAP-MLLM](https://drive.google.com/file/d/1VzNLzzjqEfVcpWDT-3m6O71CAImoD8qE/view?usp=sharing)

* **4-Concept Dataset**: We provide our 4-concept database and the corresponding evaluation dataset as `query-image-4-concepts`.

### Evaluation
Performance is reported by retrieving concepts for each sample from the respective `*.json` files located in the `database_tot` directory. The concepts are stored in a key-value format.

For our single-concept experiments, we constructed and reported performance on a separate database for each dataset.

> **⚠️ Note**: Please be aware that merging all individual databases into a single, unified database can lead to severe retrieval noise, potentially impacting performance.