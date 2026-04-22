"""
Fine-tune LLaMA-3-8B-Instruct with QLoRA on the LLM-quality-filtered
Finance-Instruct dataset.

Inputs  (local):
  data/quality_scored.csv

Outputs (local):
  adapter/   : LoRA adapter weights + tokenizer
"""
import os
os.environ['PYTHONUTF8'] = '1'

import os, torch
import numpy as np
import pandas as pd
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import login

np.random.seed(42)

login(token="hf_YOUR_TOKEN_HERE")   # replace with your token

os.makedirs("adapter", exist_ok=True)
DATA_PATH = "data/quality_scored.csv"
SAVE_DIR  = "adapter"

df_scored       = pd.read_csv(DATA_PATH)
df_model2_train = df_scored[df_scored['quality_score'] >= 3].reset_index(drop=True)
print(f"Training rows: {len(df_model2_train):,}")

#Format prompts
def format_row(row):
    return (
        '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
        + str(row['user']).strip()
        + '\n\nPlease provide a detailed and thorough explanation.'
        + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        + str(row['assistant']).strip()
        + '<|eot_id|>'
    )

df_model2_train['text'] = df_model2_train.apply(format_row, axis=1)

#train/val split
df_m2_train, df_m2_val = train_test_split(
    df_model2_train[['text']], test_size=0.10, random_state=42
)
train_dataset = Dataset.from_pandas(df_m2_train.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(df_m2_val.reset_index(drop=True))
print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

#Load base model with 4-bit quantization
print("\nLoading LLaMA-3-8B-Instruct (4-bit)...")
base_pipeline = transformers.pipeline(
    'text-generation',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    model_kwargs={
        'torch_dtype': torch.bfloat16,
        'quantization_config': {'load_in_4bit': True},
        'low_cpu_mem_usage': True,
    },
    device_map='auto',
)
model_m2     = base_pipeline.model
tokenizer_m2 = base_pipeline.tokenizer
tokenizer_m2.pad_token    = tokenizer_m2.eos_token
tokenizer_m2.padding_side = 'right'
print(f"Memory: {model_m2.get_memory_footprint()/1e9:.2f} GB")

#Attach LoRA adapter
model_m2 = prepare_model_for_kbit_training(model_m2)
model_m2 = get_peft_model(model_m2, LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
))
model_m2.print_trainable_parameters()

#Train
print("\nStarting training...")
trainer = SFTTrainer(
    model=model_m2,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir='./adapter_checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,     # effective batch = 16
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        report_to='none',
        dataset_text_field='text',
        max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    ),
)
trainer.train()

#Save LoRA adapter locally
model_m2.save_pretrained(SAVE_DIR)
tokenizer_m2.save_pretrained(SAVE_DIR)
print(f"\nLoRA adapter saved to {SAVE_DIR}/")
print("Files saved:")
for f in os.listdir(SAVE_DIR):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    print(f"  {f:40s} {size/1e6:.1f} MB")
