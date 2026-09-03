import sys

print("=" * 70)
print("GradeSense Qwen environment check")
print("=" * 70)

print("Python:", sys.version)

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Dedicated CUDA device memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                2,
            ),
            "GB",
        )
except Exception as exc:
    print("PyTorch ERROR:", repr(exc))

try:
    import transformers
    print("Transformers:", transformers.__version__)
except Exception as exc:
    print("Transformers ERROR:", repr(exc))

try:
    import accelerate
    print("Accelerate:", accelerate.__version__)
except Exception as exc:
    print("Accelerate ERROR:", repr(exc))

try:
    import bitsandbytes
    print("BitsAndBytes:", bitsandbytes.__version__)
except Exception as exc:
    print("BitsAndBytes ERROR:", repr(exc))

try:
    import qwen_vl_utils
    print("qwen-vl-utils: OK")
except Exception as exc:
    print("qwen-vl-utils ERROR:", repr(exc))

print("=" * 70)
