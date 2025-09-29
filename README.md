# LLM-Political-Inference

This repository contains the source code and dataset used in the study: **LLMs Can Infer Hidden Political Alignment of Online Users from General Conversations**


**Authors**:  
Byunghwee Lee<sup>1,2,+</sup>, Sangyeon Kim<sup>2,+</sup>, Filippo Menczer<sup>2</sup>, Yong-Yeol Ahn<sup>1,2,\*</sup>, Haewoon Kwak<sup>2,\*</sup>, Jisun An<sup>2,\*</sup>  

<sup>1</sup> <sub>School of Data Science, Charlottesville, Virginia, USA.</sub>
<sup>2</sup> <sub>Center for Complex Networks and Systems Research, Luddy School of Informatics, Computing, and Engineering, Indiana University, Bloomington, Indiana, USA.</sub>

<sup>+</sup> <sub>These authors contributed equally to this work. </sub>

<sup>\*</sup> <sub>Corresponding authors</sub>


## Introduction
 - This repository provides the source code necessary for reproducing the results presented in the paper.  
- The core implementation of experimental results is found in **`src/Main_results.ipynb`**.  


## Installation

Installation using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/ByunghweeLee-IU/LLM-political-inference.git
cd LLM-political-inference
conda create -y --name llm-inference python=3.8
conda activate llm-inference
pip install -r requirements.txt
```

## System requirements
* **Software dependencies**:
  * Supported platforms: MacOS and Ubuntu (with Python 3.8)
  * See requirements.txt for a complete list of necessary libraries.

* **Tested Versions** 
   * The following libraries have been tested with Python 3.8 or higher:
     * `torch = 2.2.1`
     * `sentence-transformers = 2.6.0`
     * `pandas = 2.2.2`
     * `numpy = 1.24.3`
     * See `requirements.txt` for full list of necessary libraries. 


## Quickstart
- Before opening `main_result.ipynb`, download the preprocessed DDO and Reddit datasets.

  - Option A (recommended): run from the repository root
    ```bash
    python src/download_data.py
    ```

  - Option B: download the files manually from the [Hugging Face dataset page](https://huggingface.co/datasets/Byunghwee/llm-inference-data)

- After downloading the dataest, run the following code in terminal.
  ```bash 
  jupyter notebook
  ```
  * Select `llm-inference` kernel in the jupyter Notebook.
  * Open `main_result.ipynb` 

## Hardware requirements

* A GPU is recommended for the inference process.
* The inference results of the **Llama-3.1-8B** model were obtained using an **NVIDIA A100 80GB PCIe** GPU.


## License
  * This project is licensed under the **MIT License** - see the `LICENSE` file for details.
