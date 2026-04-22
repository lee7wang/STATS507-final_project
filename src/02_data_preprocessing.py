"""
Load, clean, and LLM-score Finance-Instruct-500k to build the training set.
"""

import os
os.environ['PYTHONUTF8'] = '1'

import re, asyncio
import numpy as np
import pandas as pd
import anthropic
from datasets import load_dataset
from huggingface_hub import login

np.random.seed(42)
# replace with your token
login(token="hf_YOUR_TOKEN_HERE")          
os.makedirs("data", exist_ok=True)
SAVE_PATH = "data/quality_scored.csv"

print("Loading Finance-Instruct-500k...")
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k")
df      = dataset['train'].to_pandas()
print(f"Loaded: {len(df):,} rows")

def contains_chinese(text):
    if not isinstance(text, str): return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

chinese_mask = (
    df['user'].apply(contains_chinese) |
    df['assistant'].apply(contains_chinese) |
    df['system'].apply(contains_chinese)
)
df_clean = df[~chinese_mask].reset_index(drop=True)
print(f"After removing Chinese: {len(df_clean):,} rows")

df_clean['system_stripped'] = df_clean['system'].str.strip()
length_mask = (
    (df_clean['user'].str.strip().str.len() > 20) &
    (df_clean['assistant'].str.strip().str.len() > 50)
)
df_clean = df_clean[length_mask].reset_index(drop=True)
print(f"After length filter: {len(df_clean):,} rows")

df_train     = df_clean[df_clean['system_stripped'] == ''].reset_index(drop=True)
df_train_sys = df_clean[df_clean['system_stripped'].str.len() >= 5].reset_index(drop=True)
print(f"Without system prompt: {len(df_train):,} rows")
print(f"With system prompt   : {len(df_train_sys):,} rows")

os.environ['ANTHROPIC_API_KEY'] = "your-anthropic-key"   # replace

TARGET_VALID      = 5000
BATCH_SIZE        = 1000
MAX_ROWS          = 50000
MAX_CONCURRENT    = 10
QUALITY_THRESHOLD = 3


async def score_row_async(client, user_text, assistant_text):
    prompt = f"""You are evaluating a finance Q&A pair for LLM training quality.
Score it from 1 to 5:
- 1: Unusable (orphan entry, references missing text, gibberish, off-topic)
- 2: Poor (vague, inaccurate, or heavily incomplete)
- 3: Acceptable (correct but shallow or generic)
- 4: Good (accurate, reasonably detailed, on-topic finance)
- 5: Excellent (accurate, detailed, clear financial reasoning)

QUESTION: {user_text[:400]}
ANSWER: {assistant_text[:400]}

Reply with ONLY a single integer 1, 2, 3, 4, or 5. Nothing else."""
    try:
        response = await client.messages.create(
            model='claude-sonnet-4-6', max_tokens=5,
            messages=[{'role': 'user', 'content': prompt}]
        )
        score = int(response.content[0].text.strip())
        return score if 1 <= score <= 5 else -1
    except:
        return -1


async def score_batch_async(client, batch_df):
    semaphore  = asyncio.Semaphore(MAX_CONCURRENT)
    completed  = 0
    total      = len(batch_df)
    results    = [None] * total

    async def score_with_limit(idx, row):
        nonlocal completed
        async with semaphore:
            result = await score_row_async(client, row['user'], row['assistant'])
        results[idx] = result
        completed += 1
        if completed % 50 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} ({completed/total*100:.0f}%)", flush=True)

    tasks = [score_with_limit(i, row) for i, (_, row) in enumerate(batch_df.iterrows())]
    await asyncio.gather(*tasks)
    return results


async def run_scoring(df_train):
    client = anthropic.AsyncAnthropic()

    if os.path.exists(SAVE_PATH):
        df_previous = pd.read_csv(SAVE_PATH)
        total_valid = (df_previous['quality_score'] >= QUALITY_THRESHOLD).sum()
        all_scored  = [df_previous]
        start_idx   = len(df_previous)
        print(f"Resuming: {start_idx:,} scored, {total_valid:,} valid")
    else:
        all_scored  = []
        total_valid = 0
        start_idx   = 0

    df_candidate = df_train.head(MAX_ROWS).sample(frac=1, random_state=42).reset_index(drop=True)

    for batch_start in range(start_idx, MAX_ROWS, BATCH_SIZE):
        batch  = df_candidate.iloc[batch_start:batch_start + BATCH_SIZE].copy()
        scores = await score_batch_async(client, batch)
        batch['quality_score'] = scores

        valid_in_batch = (batch['quality_score'] >= QUALITY_THRESHOLD).sum()
        total_valid   += valid_in_batch
        all_scored.append(batch)

        pd.concat(all_scored, ignore_index=True).to_csv(SAVE_PATH, index=False)
        print(f"Batch {batch_start//BATCH_SIZE+1} done — valid: {total_valid:,}/{TARGET_VALID:,}")

        if total_valid >= TARGET_VALID:
            print("Target reached!")
            break

    return pd.concat(all_scored, ignore_index=True)


def main():
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, run_scoring(df_train))
            df_all = future.result()
    except RuntimeError:
        df_all = asyncio.run(run_scoring(df_train))

    df_valid  = df_all[df_all['quality_score'] >= QUALITY_THRESHOLD]
    df_model2 = df_valid.head(TARGET_VALID)[['system', 'user', 'assistant']].reset_index(drop=True)
    df_model2.to_parquet("data/model2_train.parquet", index=False)

    print(f"Valid rate: {len(df_valid)/len(df_all)*100:.1f}% — {len(df_model2):,} rows saved.")


if __name__ == "__main__":
    main()
