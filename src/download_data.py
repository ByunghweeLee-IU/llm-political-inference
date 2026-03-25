#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

HF_DATASET_REPO = "Byunghwee/llm-inference-data"  
TARGET_DIR = Path(__file__).resolve().parents[1] / "data" / "llm-inference-data"

def main():
    token = os.getenv("HF_TOKEN", None)  
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Downloading from hf://{HF_DATASET_REPO} -> {TARGET_DIR}")

    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,  
            token=token,
        )
    except Exception as e:
        print("\n[error] Failed to download from Hugging Face.")
        if token is None:
            print("        Private repo requires an HF_TOKEN.")
            print("        1) Generate a token from https://huggingface.co/settings/tokens")
            print("        2) export HF_TOKEN=YOUR_TOKEN  (Windows: set HF_TOKEN=...)")
        print(f"\n[details] {e}\n")
        sys.exit(1)

    # 간단한 검증: 예상 파일 중 하나 체크 (원하면 더 추가)
    expected_any = [
        TARGET_DIR / "df_reddit_hf.parquet",
        TARGET_DIR / "df_ddo_hf.parquet",
    ]
    if not any(p.exists() for p in expected_any):
        print("[warn] Downloaded, but expected parquet files not found.")
        print(f"       Check contents in: {TARGET_DIR}")
    else:
        print("[done] Data available at:", TARGET_DIR)

if __name__ == "__main__":
    main()

