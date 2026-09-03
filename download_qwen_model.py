import os
from pathlib import Path

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "Qwen2.5-VL-3B-Instruct"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GradeSense - Qwen2.5-VL-3B-Instruct downloader")
print("=" * 70)
print()
print("Model:")
print(MODEL_ID)
print()
print("Destination:")
print(MODEL_DIR)
print()
print("Download timeout: 120 seconds")
print("Concurrent downloads: 2")
print()
print("Already completed files will be reused.")
print("Do NOT delete the .cache folder.")
print("=" * 70)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(MODEL_DIR),
    max_workers=2,
    resume_download=True,
)

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print()
print("Model location:")
print(MODEL_DIR)
