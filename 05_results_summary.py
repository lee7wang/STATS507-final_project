import json
import pandas as pd

RESULTS_DIR = "results"

#Load results
with open(f'{RESULTS_DIR}/perplexity_results.json')  as f: ppl  = json.load(f)
with open(f'{RESULTS_DIR}/bert_score_results.json')  as f: bert = json.load(f)
df_econ = pd.read_csv(f'{RESULTS_DIR}/econ_eval_results.csv')
df_gen  = pd.read_csv(f'{RESULTS_DIR}/general_eval_results.csv')

#Print consolidated table
print("=" * 60)
print("COMPLETE EVALUATION SUMMARY")
print("Project: Enhancing Financial Reasoning via QLoRA Fine-Tuning")
print("=" * 60)
print(f"{'Metric':<35}  {'Base':>8}  {'Model2':>8}  {'Delta':>8}")
print("-" * 60)

rows = [
    ("Perplexity (val set)",         ppl['base_ppl'],                ppl['m2_ppl']),
    ("BERTScore F1 (EconLogicQA)",   bert['base_f1'],                bert['m2_f1']),
    ("EconLogicQA Avg Accuracy",     df_econ['base_accuracy'].mean(),df_econ['m2_accuracy'].mean()),
    ("EconLogicQA Avg Reasoning",    df_econ['base_reasoning'].mean(),df_econ['m2_reasoning'].mean()),
    ("EconLogicQA Hallucination",    df_econ['base_halluc'].mean(),  df_econ['m2_halluc'].mean()),
    ("General Domain Accuracy",      df_gen['base_accuracy'].mean(), df_gen['m2_accuracy'].mean()),
    ("General Domain Hallucination", df_gen['base_halluc'].mean(),   df_gen['m2_halluc'].mean()),
]

for label, base_val, m2_val in rows:
    delta = m2_val - base_val
    arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '=')
    print(f"{label:<35}  {base_val:>8.4f}  {m2_val:>8.4f}  {arrow}{delta:>+7.4f}")

print("-" * 60)

econ_m2_w  = (df_econ['winner'] == 'model2').sum()
econ_bas_w = (df_econ['winner'] == 'base').sum()
econ_ties  = (df_econ['winner'] == 'tie').sum()
gen_m2_w   = (df_gen['winner'] == 'model2').sum()
gen_bas_w  = (df_gen['winner'] == 'base').sum()

print(f"{'EconLogicQA Win Rate':<35}  Base:{econ_bas_w:>2}W  M2:{econ_m2_w:>2}W  Ties:{econ_ties:>2}")
print(f"{'General Domain Win Rate':<35}  Base:{gen_bas_w:>2}W  M2:{gen_m2_w:>2}W")
print("=" * 60)

forgetting = "NOT detected ✓" if gen_m2_w >= gen_bas_w else "DETECTED ✗"
print(f"\nCatastrophic forgetting : {forgetting}")
print(f"Val set perplexity delta: {ppl['improvement_pct']:+.1f}%")
print(f"BERTScore improvement   : {bert['delta']:+.4f}")