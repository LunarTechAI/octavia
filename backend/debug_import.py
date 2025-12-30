
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
    print(f"Torch: {torch.__version__}")
    import torchvision
    print(f"TorchVision: {torchvision.__version__}")
    import torchaudio
    print(f"TorchAudio: {torchaudio.__version__}")
    from transformers import pipeline
    print("Transformers imported")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
