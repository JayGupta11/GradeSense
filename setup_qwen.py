"""
Install only the packages required by the GradeSense Qwen OCR module.

Run this while the existing GradeSense `haesenv` environment is active:

    python setup_qwen.py

This script intentionally does NOT reinstall PyTorch.
"""

import subprocess
import sys

PACKAGES = [
    "transformers>=4.57.0,<6",
    "accelerate>=1.2.0",
    "bitsandbytes>=0.46.1",
    "qwen-vl-utils>=0.0.14",
    "Pillow>=10.0",
]

print("=" * 70)
print("GradeSense Qwen OCR dependency setup")
print("=" * 70)
print("Python:", sys.version)
print()

for package in PACKAGES:
    print(f"Installing/upgrading: {package}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", package]
    )

print("\nSetup complete.")
print("PyTorch was NOT changed.")
print("Next: python test_environment.py")
