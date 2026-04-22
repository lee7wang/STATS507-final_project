"""
01_setup.py
===========
Environment setup and package installation.
Run this first in every new Colab session before anything else.
"""

import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers", "datasets", "peft", "trl",
    "bitsandbytes", "accelerate", "sentencepiece",
    "anthropic", "bert-score", "scikit-learn"
])

import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")

#HuggingFace login
from huggingface_hub import login
login(token="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")   # replace with your token

import os
os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("adapter", exist_ok=True)
print("Setup complete!")
print("Directory structure:")
print("  data/    — training data and quality scores")
print("  results/ — evaluation outputs")
print("  adapter/ — saved LoRA adapter weights")
