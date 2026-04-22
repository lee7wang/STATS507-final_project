"""
Load fine-tuned model and run comprehensive evaluation against base LLaMA-3.

Inputs  (local):
  - data/quality_scored.csv
  - adapter/               : LoRA adapter weights

Evaluations:
  1. EconLogicQA benchmark 
  2. Perplexity            
  3. BERTScore             
  4. General domain        
  5. Claude-as-Judge       
"""

import json, random, os, torch, math
import numpy as np
import pandas as pd
import anthropic
import transformers
from peft import PeftModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from bert_score import score as bert_score_fn
from huggingface_hub import login
from tqdm import tqdm

np.random.seed(42)
random.seed(42)

login(token="hf_YOUR_TOKEN_HERE")               # replace
os.environ['ANTHROPIC_API_KEY'] = "your-anthropic-key"   # replace
client = anthropic.Anthropic()

#Local paths
os.makedirs("results", exist_ok=True)
DATA_PATH   = "data/quality_scored.csv"
ADAPTER_DIR = "adapter"
RESULTS_DIR = "results"

#Load base model
print("Loading base LLaMA-3-8B-Instruct...")
base_pipe = transformers.pipeline(
    'text-generation',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    model_kwargs={
        'torch_dtype': torch.bfloat16,
        'quantization_config': {'load_in_4bit': True},
        'low_cpu_mem_usage': True,
    },
    device_map='auto',
)
base_model     = base_pipe.model
base_tokenizer = base_pipe.tokenizer
print("Base model loaded!")

#Load fine-tuned model (LoRA adapter)
print("Loading fine-tuned model from adapter/...")
model_m2     = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
tokenizer_m2 = base_tokenizer
print("Fine-tuned model loaded!")

#Helpers
def generate_answer(question, model, tokenizer, max_new_tokens=400):
    prompt = (
        '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
        + question.strip()
        + '\n\nPlease provide a detailed and thorough explanation.'
        + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    )
    inputs = tokenizer(prompt, return_tensors='pt',
                       add_special_tokens=False).to(model.device)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids('<|eot_id|>'),
        )
    input_length = inputs['input_ids'].shape[1]
    return tokenizer.decode(output_ids[0][input_length:],
                            skip_special_tokens=True).strip()


def judge_pair(question, base_ans, m2_ans, domain="finance"):
    swapped  = random.random() > 0.5
    ans_a    = m2_ans   if swapped else base_ans
    ans_b    = base_ans if swapped else m2_ans
    label_a  = 'model2' if swapped else 'base'
    label_b  = 'base'   if swapped else 'model2'

    domain_note = ("This is a financial/economic reasoning question."
                   if domain == "finance"
                   else "This is a general knowledge question (NOT finance).")

    prompt = f"""You are an expert evaluating two AI answers. {domain_note}

QUESTION: {question}

ANSWER A:
{ans_a[:600]}

ANSWER B:
{ans_b[:600]}

Evaluate: accuracy, reasoning, completeness, hallucination (5=none,1=many).
Reply ONLY valid JSON no markdown:
{{"winner":"A" or "B" or "tie","reasoning":"one sentence",
  "score_a":{{"accuracy":1-5,"reasoning":1-5,"completeness":1-5,"hallucination":1-5}},
  "score_b":{{"accuracy":1-5,"reasoning":1-5,"completeness":1-5,"hallucination":1-5}}}}"""

    response = client.messages.create(
        model='claude-sonnet-4-6', max_tokens=400,
        messages=[{'role': 'user', 'content': prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'): raw = raw[4:]
    try:
        result     = json.loads(raw.strip())
        winner     = label_a if result['winner']=='A' else (
                     label_b if result['winner']=='B' else 'tie')
        score_m2   = result['score_a'] if swapped else result['score_b']
        score_base = result['score_b'] if swapped else result['score_a']
        return {'winner': winner, 'reasoning': result['reasoning'],
                'score_base': score_base, 'score_m2': score_m2}
    except:
        print("JSON parse error — skipping question")
        return None


# 1. EconLogicQA
print("\n" + "="*50)
print("1. EconLogicQA Evaluation (n=30)")
print("="*50)
econ    = load_dataset("yinzhu-quan/econ_logic_qa", split="test")
indices = random.sample(range(len(econ)), 30)
sample  = econ.select(indices)

test_questions, gold_answers = [], []
for row in sample:
    q = (f"{row['Question']}\n\nEvents to arrange:\n"
         f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}\n\n"
         f"Arrange A, B, C, D in the correct logical order.")
    test_questions.append(q)
    gold_answers.append(row['Answer'])

base_answers = [generate_answer(q, base_model, base_tokenizer) for q in tqdm(test_questions, desc="Base")]
m2_answers   = [generate_answer(q, model_m2,   tokenizer_m2)   for q in tqdm(test_questions, desc="M2")]

with open(f'{RESULTS_DIR}/econ_answers.json', 'w') as f:
    json.dump({'questions': test_questions, 'gold_answers': gold_answers,
               'base': base_answers, 'model2': m2_answers}, f)

import re
def extract_seq(text):
    m = re.search(r'\b([ABCD])[,\s]+([ABCD])[,\s]+([ABCD])[,\s]+([ABCD])\b', text.upper())
    return ', '.join(m.groups()) if m else None

base_correct = sum(1 for p, g in zip(map(extract_seq, base_answers), gold_answers) if p == g.strip())
m2_correct   = sum(1 for p, g in zip(map(extract_seq, m2_answers),   gold_answers) if p == g.strip())
print(f"Exact match — Base: {base_correct}/30 ({base_correct/30*100:.1f}%) | "
      f"Model2: {m2_correct}/30 ({m2_correct/30*100:.1f}%)")

econ_results = []
for q, gold, b, m2 in zip(test_questions, gold_answers, base_answers, m2_answers):
    j = judge_pair(q, b, m2, "finance")
    if j:
        econ_results.append({
            'question':       q[:80], 'gold': gold, 'winner': j['winner'],
            'reasoning':      j['reasoning'],
            'base_accuracy':  j['score_base']['accuracy'],
            'm2_accuracy':    j['score_m2']['accuracy'],
            'base_reasoning': j['score_base']['reasoning'],
            'm2_reasoning':   j['score_m2']['reasoning'],
            'base_halluc':    j['score_base']['hallucination'],
            'm2_halluc':      j['score_m2']['hallucination'],
        })

df_econ = pd.DataFrame(econ_results)
df_econ.to_csv(f'{RESULTS_DIR}/econ_eval_results.csv', index=False)
print(f"Win rate — Model2: {(df_econ['winner']=='model2').sum()} | "
      f"Base: {(df_econ['winner']=='base').sum()} | "
      f"Ties: {(df_econ['winner']=='tie').sum()}")
print(f"Avg accuracy  — Base: {df_econ['base_accuracy'].mean():.2f} | "
      f"Model2: {df_econ['m2_accuracy'].mean():.2f}")
print(f"Avg reasoning — Base: {df_econ['base_reasoning'].mean():.2f} | "
      f"Model2: {df_econ['m2_reasoning'].mean():.2f}")


# 2. Perplexity
print("\n" + "="*50)
print("2. Perplexity on Quality-Filtered Val Set")
print("="*50)

df_scored = pd.read_csv(DATA_PATH)
df_valid  = df_scored[df_scored['quality_score'] >= 3].reset_index(drop=True)
_, df_val = train_test_split(df_valid, test_size=0.10, random_state=42)

def fmt(row):
    return ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
            + str(row['user']).strip()
            + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            + str(row['assistant']).strip() + '<|eot_id|>')

eval_texts = df_val.apply(fmt, axis=1).tolist()
print(f"Val set: {len(eval_texts)} rows")

def compute_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss = total_tokens = 0
    for text in texts:
        inp = tokenizer(text, return_tensors='pt', max_length=max_length,
                        truncation=True).to(model.device)
        with torch.no_grad():
            loss = model(**inp, labels=inp['input_ids']).loss.item()
        n = inp['input_ids'].shape[1]
        total_loss += loss * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)

ppl_base = compute_perplexity(base_model, base_tokenizer, eval_texts)
ppl_m2   = compute_perplexity(model_m2,   tokenizer_m2,   eval_texts)
improv   = (ppl_base - ppl_m2) / ppl_base * 100
print(f"Base PPL: {ppl_base:.4f} | Model2 PPL: {ppl_m2:.4f} | Δ: {improv:+.1f}%")

with open(f'{RESULTS_DIR}/perplexity_results.json', 'w') as f:
    json.dump({'n_val': len(eval_texts), 'base_ppl': round(ppl_base, 4),
               'm2_ppl': round(ppl_m2, 4), 'improvement_pct': round(improv, 2)}, f)


# 3. BERTScore
print("\n" + "="*50)
print("3. BERTScore (semantic similarity to gold answers)")
print("="*50)

_, _, F_base = bert_score_fn(base_answers, gold_answers, lang='en', verbose=False)
_, _, F_m2   = bert_score_fn(m2_answers,   gold_answers, lang='en', verbose=False)
delta_bert   = float(F_m2.mean() - F_base.mean())
print(f"Base F1: {F_base.mean():.4f} | Model2 F1: {F_m2.mean():.4f} | Δ: {delta_bert:+.4f}")

with open(f'{RESULTS_DIR}/bert_score_results.json', 'w') as f:
    json.dump({'base_f1': round(float(F_base.mean()), 4),
               'm2_f1':   round(float(F_m2.mean()), 4),
               'delta':   round(delta_bert, 4)}, f)


# 4. General domain — catastrophic forgetting
print("\n" + "="*50)
print("4. General Domain — Catastrophic Forgetting Check (n=10)")
print("="*50)

general_questions = [
    "What causes seasons on Earth?",
    "Explain how vaccines work in the immune system.",
    "What is the difference between DNA and RNA?",
    "How does a combustion engine work?",
    "What were the main causes of World War I?",
    "Explain the theory of evolution by natural selection.",
    "What is the difference between weather and climate?",
    "How does the internet route data packets?",
    "What is the Pythagorean theorem and when is it used?",
    "Explain how photosynthesis works in plants.",
]
base_general = [generate_answer(q, base_model, base_tokenizer)
                for q in tqdm(general_questions, desc="Base")]
m2_general   = [generate_answer(q, model_m2, tokenizer_m2)
                for q in tqdm(general_questions, desc="M2")]

with open(f'{RESULTS_DIR}/general_answers.json', 'w') as f:
    json.dump({'questions': general_questions,
               'base': base_general, 'model2': m2_general}, f)

gen_results = []
for q, b, m2 in zip(general_questions, base_general, m2_general):
    j = judge_pair(q, b, m2, "general")
    if j:
        gen_results.append({
            'question':      q,
            'winner':        j['winner'],
            'base_accuracy': j['score_base']['accuracy'],
            'm2_accuracy':   j['score_m2']['accuracy'],
            'base_halluc':   j['score_base']['hallucination'],
            'm2_halluc':     j['score_m2']['hallucination'],
        })

df_gen = pd.DataFrame(gen_results)
df_gen.to_csv(f'{RESULTS_DIR}/general_eval_results.csv', index=False)

m2_gen_wins   = (df_gen['winner'] == 'model2').sum()
base_gen_wins = (df_gen['winner'] == 'base').sum()
print(f"Win rate — Model2: {m2_gen_wins} | Base: {base_gen_wins} | "
      f"Ties: {(df_gen['winner']=='tie').sum()}")
if base_gen_wins > m2_gen_wins + 2:
    print("Conclusion: Catastrophic forgetting DETECTED")
else:
    print("Conclusion: No catastrophic forgetting detected")

print("\n" + "="*50)
print("All evaluations complete!")
print(f"Results saved to {RESULTS_DIR}/")
print("="*50)
