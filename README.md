**University of Michigan \| Li Wang**

------------------------------------------------------------------------

## Requirements

-   HuggingFace account with access to
    [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
-   Anthropic API key

------------------------------------------------------------------------

Run the scripts in `src/` in order.

### Step 1 — Install packages and authenticate

```         
src/01_setup.py
```

Fill in your HuggingFace token where indicated.

### Step 2 — Preprocess data and run LLM quality scoring

```         
src/02_data_preprocessing.py
```

Fill in your Anthropic API key where indicated.

> **To skip this step:** Download `quality_scored.csv` from Google drive
> <https://drive.google.com/file/d/1iIbxy0uuvPkzO6jQfbV6f-RSBNujIylI/view?usp=sharingCopy>.
> Move it to `/data` folder.

### Step 3 — Fine-tune the model

```         
src/03_finetune_model2.py
```

Trains LLaMA-3-8B-Instruct with QLoRA on the quality-filtered data.

> **To skip this step:** Download the pre-trained LoRA adapter from
> Google Drive:
> <https://drive.google.com/drive/folders/1LJs4YRZsLCtlshZOm_ZEJzhq9QZUvVma?usp=sharing>.
> Unzip the file and place it at `MyDrive/llama3-finance-lora-m2/final/`
> on Google Drive.

### Step 4 — Run evaluation

```         
src/04_evaluation.py
```

Fill in your Anthropic API key. Runs EconLogicQA benchmark, perplexity,
BERTScore, and general domain catastrophic forgetting evaluation.

### Step 5 — View results summary

```         
src/05_results_summary.py
```

------------------------------------------------------------------------

## Results

Results from our experimental run are saved in `results/`:

| File                       | Contents                                    |
|----------------------------|---------------------------------------------|
| `econ_eval_results.csv`    | EconLogicQA Claude-as-Judge pairwise scores |
| `general_eval_results.csv` | General domain forgetting check scores      |
| `perplexity_results.json`  | Perplexity: Base=1.9857, Model2=1.987       |
| `bert_score_results.json`  | BERTScore F1: Base=0.8522, Model2=0.8698    |

------------------------------------------------------------------------
