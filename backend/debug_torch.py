
import os
import sys

# Try adding Library/bin to DLL search path
try:
    os.add_dll_directory(r"C:\Users\tatev\Anaconda3\Library\bin")
    print("Added Library/bin to DLL path")
except Exception as e:
    print(f"Could not add DLL dir: {e}")

try:
    import torch
    print(f"Torch imported! Version: {torch.__version__}")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
