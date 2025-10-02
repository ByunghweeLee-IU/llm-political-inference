
# infer_party_llama_langchain_min.py
# Minimal single-text inference using LangChain + HF Transformers (Llama).
# - Uses ChatPromptTemplate | HuggingFacePipeline chain
# - Keeps messages as in your Llama prompt (with JSON examples)
# - Returns ONLY {"party", "confidence"}

import os
import re
import json
from typing import Optional, Dict, Any, List

import torch
from pydantic import BaseModel, Field
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate


# ---------- Utilities ----------
class PartyInference(BaseModel):
    party: str = Field(description="party: Democratic or Republican")
    confidence: int = Field(description="confidence: from 1 to 5")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first {...} JSON object from text and parse it."""
    if not isinstance(text, str):
        return None
    # In case the model echoes the prompt, drop the leading guidance part used in your code:
    text = "".join(text.split("Now, please classify the following text:")[1:])
    m = re.search(r'\{[\s\S]*?\}', text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _login_hf(hf_token: Optional[str] = None) -> str:
    """Login to Hugging Face with precedence: arg -> env LLAMA_API_KEY -> ../data/credentials_HF.txt"""
    token = hf_token or os.getenv("LLAMA_API_KEY")
    if not token:
        cred_path = os.path.join(os.path.dirname(__file__), "../data/credentials_HF.txt")
        if os.path.exists(cred_path):
            token = open(cred_path, "r").read().strip()
    if not token:
        raise RuntimeError("HF token not found. Set LLAMA_API_KEY or place it in ../data/credentials_HF.txt")
    login(token)
    return token

# ---------- Core single-text inference (LangChain flow) ----------
def infer_party_llama_langchain(
    input_text: str,
    *,
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct", #"meta-llama/Llama-3.2-3B-Instruct" 
    hf_token: Optional[str] = None,
    hf_home: Optional[str] = None,
    device: Optional[int] = 0,          # HF pipeline uses int device index (e.g., 0) or -1 for CPU
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> Dict[str, object]:
    """Run a single inference using LangChain (ChatPromptTemplate | HuggingFacePipeline).
    Returns ONLY {'party': <str or None>, 'confidence': <int or None>}.
    """
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("input_text must be a non-empty string.")

    # Optional cache dir for HF
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    # Login to HF
    _login_hf(hf_token)

    # Device: if cuda available and device is None, default to 0; else CPU -> -1
    if device is None:
        if torch.cuda.is_available():
            device = 0
            use_mps = False
        elif torch.backends.mps.is_available():
            device = -1   # pipeline용은 CPU (-1)로 두고, 직접 model.to("mps") 해줌
            use_mps = True
        else:
            device = -1
            use_mps = False
    else:
        use_mps = False

    # Load tokenizer/model and build a text-generation pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if len(model.get_input_embeddings().weight) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if use_mps:
        model = model.to("mps")
        print("Using MPS (Apple GPU)")

    txt_gen = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        device=device
    )

    # Wrap into LangChain HF pipeline
    lc_model = HuggingFacePipeline(pipeline=txt_gen)

    # Messages (kept as in your Llama script with JSON format + examples)
    messages = [
        ("system", """You are a helpful assistant. Based on the following debater's text, 
infer whether the debater's position aligns more with the Republican Party or the Democratic Party.
Provide the answer with the confidence level of your answer.
You must respond only with valid JSON in the following format:

{{
  "party": "Democratic" or "Republican",
  "confidence": integer between 1 and 5
}}

Examples:

Text: "I believe in universal healthcare."
Response:
{{
  "party": "Democratic",
  "confidence": 4
}}

Text: "I support lower taxes for businesses."
Response:
{{
  "party": "Republican",
  "confidence": 5
}}"""),
        ("user", "Now, please classify the following text: {input}"),
        ("system","")
    ]

    # LangChain prompt and chain
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | lc_model

    # Invoke with the user's text
    response = chain.invoke({"input": input_text})

    # Extract JSON and validate
    parsed = _extract_json(response) or {}
    try:
        validated = PartyInference(**parsed)
        return validated.dict()
    except Exception:
        return {"party": None, "confidence": None}

# ---------- Example main ----------
if __name__ == "__main__":
    sample = "I support expanding access to affordable healthcare and stricter climate policies."
    out = infer_party_llama_langchain(
        sample,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        hf_home="../data/",
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=256,
        temperature=0.2,
    )
    print(out)
