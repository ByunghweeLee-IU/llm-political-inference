import os
import json
from pathlib import Path as _Path
from typing import Optional, Dict

import openai

def _load_key_into_env(api_key: Optional[str] = None) -> None:
    """
    Load API key into environment, following the same pattern as the research code:
      1) If api_key arg is provided, use it.
      2) Else, try OPENAI_API_KEY env var.
      3) Else, read ../data/credentials_openai.txt (first line) and set env.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        cred_path = _Path(__file__).resolve().parent.parent / "data" / "credentials_openai.txt"
        if cred_path.exists():
            os.environ["OPENAI_API_KEY"] = cred_path.read_text().strip()

    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        raise RuntimeError("OpenAI API key not found. Set env or place it in ../data/credentials_openai.txt")

def infer_party(
    input_string: str,
    *,
    model_version: str = "gpt-4o-2024-08-06",
    api_key: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run inference for a single debate text.
    Returns ONLY {'party': <str or None>, 'confidence': <int/str or None>}.
    """
    if not isinstance(input_string, str) or not input_string.strip():
        raise ValueError("input_string must be a non-empty string.")

    # Load key into env exactly like the original main()
    _load_key_into_env(api_key=api_key)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Original client creation style
    client = openai.OpenAI(api_key=openai.api_key)

    # Messages block MUST remain exactly the same for reproduction
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 
         """Based on the following text from a debate, infer whether the debater's position aligns more with the Republican Party or the Democratic Party.
         Respond with 'Republican' or 'Democratic'. 
         Also indicate your level of confidence for your classification on a scale from 1 to 5. Your answer must be an integer.
         Response in json format with 'party', 'confidence'.
         """},
        {"role": "assistant", "content": "Please provide your input"},
        {"role": "user", "content": "Please classify this text: " + input_string}
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model=model_version,
            messages=messages,
            response_format={"type":"json_object"}
        )
        raw = completion.choices[0].message.content
        result = json.loads(raw)

        # ONLY party and confidence returned
        return {
            "party": result.get("party"),
            "confidence": result.get("confidence"),
        }

    except Exception as e:
        # Match the spirit of the original code's failure path (append NaN),
        # but here we simply return None fields for a single-call helper.
        return {
            "party": None,
            "confidence": None,
        }

if __name__ == "__main__":
    text = "We must protect the Second Amendment while ensuring public safety."
    print(infer_party(text))

