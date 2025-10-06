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
- The core implementation of experimental results is found in **`main_result.ipynb`**.  

## System requirements
* Supported platforms: MacOS and Ubuntu (with Python 3.8)
* See requirements.txt for a complete list of necessary libraries.


## Installation

Installation of environment and dependencies using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/ByunghweeLee-IU/LLM-political-inference.git
cd LLM-political-inference
conda create -y --name llm-inference python=3.8
conda activate llm-inference
pip install -r requirements.txt
```


## Quickstart (Main results)
- Before opening `main_result.ipynb`, download the preprocessed DDO and Reddit datasets.

  - Option A (recommended): run from the repository root
    ```bash
    python src/download_data.py
    ```

  - Option B: download the files manually from the [Hugging Face dataset page](https://huggingface.co/datasets/Byunghwee/llm-inference-data) and place the `llm-inference-data` folder inside the `data` directory located at the root of the repository.

- After downloading the dataest, run the following code in terminal.
  ```bash 
  jupyter notebook
  ```
  * Select `llm-inference` kernel in the jupyter Notebook and open `main_result.ipynb` 

## (Example) Inferring political party and confidence
We provide two examples to run political party inference on a single input text:

- **OpenAI GPT-4o based inference** (`src/party_inference_gpt.py`)  
	- Run with GPT (requires OpenAI API key) 
		- ```python src/party_inference_gpt.py ```
		- Example output: ```json
{"party": "Democratic", "confidence": 4}
```

- **Llama-3.1-8B HuggingFace model inference** (`src/party_inference_llama.py`)
	-  Run with Llama (requires HuggingFace token & GPU recommended)
		- ```python src/party_inference_llama.py```
		- Example output: ```json
{"party": "Republican", "confidence": 5}
```

Both scripts will take a sample input text (e.g., *"I support expanding access to affordable healthcare and stricter climate policies."*) and return a JSON object with the predicted party (`Democratic` or `Republican`) and a confidence score (1–5).


## Project Structure

```
src/
 ├── download_data.py            # Downloading Pre-processed dataset
 ├── party_inference_gpt.py      # GPT-based single text inference
 ├── party_inference_llama.py    # Llama-based single text inference
 └── util.py                     # helper functions (metrics, utilities)
data/
main_result.ipynb
requirements.txt                 # dependencies
README.md                        # project documentation
```

## Hardware requirements

* A GPU is recommended for the inference process.
* The inference results of the **Llama-3.1-8B** model were obtained using an **NVIDIA A100 80GB PCIe** GPU.


## License
  * This project is licensed under the **MIT License** - see the `LICENSE` file for details.
