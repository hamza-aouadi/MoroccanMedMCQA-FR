# MoroccanMedMCQA-FR

Evaluation code for **MoroccanMedMCQA-FR**, the first French-language medical
multiple-choice question-answering benchmark derived from official Moroccan
medical faculty examinations.

- **Paper:** *Evaluating Large Language Models on Moroccan Medical Faculty Examinations: The MoroccanMedMCQA-FR Benchmark*, 2026 *(to appear)*
- **Dataset:** https://huggingface.co/datasets/hamzaaouadi/MoroccanMedMCQA-FR
  (DOI: [10.57967/hf/8867](https://doi.org/10.57967/hf/8867), license: CC BY-NC 4.0)
- 6,771 gold-standard questions · 31 specialties · single-answer (MCQU) + multi-answer (MCQM)
- Annotated with cognitive focus (Understanding / Reasoning), curriculum year,
  exam session, and clinical context.

> The **test split is released without correct answers** to preserve benchmark
> integrity. This repository never contains the test-split gold answers.

## Repository layout

```
.
├── question_focus/                 # cognitive-focus annotation pipeline (Understanding / Reasoning)
│   ├── annotate.py                 #   GPT-4o labeling + GPT-5.5 validation
│   ├── compare_labels.py           #   agreement + Cohen's kappa between two label files
│   ├── inputs/                     #   prompt (Fig. 5) + 279 blind validation Qs + the two doctors' labels
│   └── results/                    #   label outputs land here
│
├── evaluation/                     # run models over the benchmark and score them
│   ├── run_eval.py                 #   menu of all 36 model runs; writes predictions
│   ├── openrouter.py               #   call() for the OpenRouter models
│   ├── vertex.py                   #   call() for the Vertex models
│   ├── parser.py                   #   answer extraction (letters / NONE / unparseable)
│   ├── metrics.py                  #   Accuracy (MCQU), EMR + Hamming/Jaccard (MCQM)
│   ├── baselines.py                #   naive-baseline predictions (no model calls)
│   ├── prompts/                    #   mcqu.txt (Fig. 6) + mcqm.txt (Fig. 7)
│   ├── inputs/                     #   answer-free test set (MoroccanMedMCQA-FR_test_no_answers.json)
│   └── results/                    #   predictions + baseline CSVs land here
│
├── requirements.txt
├── CITATION.cff
└── LICENSE                         # Apache-2.0
```

Each script is run **from inside its own folder**; `inputs/` and `results/` are relative to it.

## Installation

```bash
git clone https://github.com/hamza-aouadi/MoroccanMedMCQA-FR.git
cd MoroccanMedMCQA-FR
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## API access

- **OpenRouter models** — set `OPENROUTER_API_KEY` (the runner prompts for it if unset).
- **OpenAI** (cognitive-focus annotation) — set `OPENAI_API_KEY`.
- **Vertex models** — no API key; authenticate with Application Default Credentials:
  ```bash
  gcloud auth application-default login
  ```
  Each Vertex model must be deployed first; the runner then asks for its
  OpenAI-compatible endpoint URL and an access token
  (`gcloud auth application-default print-access-token`).

```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...
```

## Usage

### Cognitive-focus annotation (Understanding / Reasoning)

```bash
cd question_focus

# label with GPT-4o (edit MODEL in the CONFIG block for the GPT-5.5 validation run)
python annotate.py

# inter-annotator agreement + Cohen's kappa (the two medical doctors)
python compare_labels.py
```

`inputs/` holds the exact labeling prompt (Fig. 5), the 279 blind validation
questions, and the two doctors' independent Understanding/Reasoning labels
(inter-annotator Cohen's kappa = 0.80).

### Evaluate a model

```bash
cd evaluation
python run_eval.py        # shows the 36-model menu; pick a number, or 0 for ALL
```

Writes `results/<label>_predictions.jsonl`, one line per question:
`{id, question_type, question_focus, speciality, pred, raw}`. No scoring is done
here (the test set is answer-free).

### Score predictions

```bash
cd evaluation
python metrics.py                                  # scores every results/*_predictions.jsonl
# or: python metrics.py results/<label>_predictions.jsonl
```

It prompts for the path to your **private answered** test JSON (the file WITH
`correct_answers`), joins by `id`, and prints per model:

- **MCQU** — Accuracy: `1` if the predicted letter equals the gold letter.
- **MCQM** — Exact Match Ratio (EMR): `1` if the predicted set equals the gold set.
- **MCQM** — Hamming Score = **Jaccard** index `|G ∩ P| / |G ∪ P|` (both sets empty → `1.0`).

`UNPARSEABLE` / `ERROR` predictions count as wrong.

### Naive baselines

```bash
cd evaluation
python baselines.py       # Random-1 / Always-A / Longest / Majority (MCQU); Random-k / All-options (MCQM)
```

Reads the answer-free test set and writes prediction CSVs to `results/`; score
them with `metrics.py` the same way as model predictions.

## Models evaluated

36 model runs across OpenRouter and Google Vertex AI — vanilla and reasoning
LLMs (GPT-5.5, GPT-4o, Claude Sonnet 5, DeepSeek-V4-Pro, Qwen3.8-27B, Grok 4.20,
Llama 4 Scout/Maverick, Mistral Medium 3.5, Gemma 4, GLM-5.2, gpt-oss 20B/120B,
Gemini 3.1 Pro / 3.5 Flash / Flash-Lite, Kimi K3) and biomedical specialized
models (MedGemma 4B/27B/1.5-4B, OpenBioLLM 8B/70B, BioMistral 7B, Phi-4). The
full list with provider and reasoning-effort settings is the `MODELS` registry
in `evaluation/run_eval.py`.

## Citation

If you use this benchmark or code, please cite:

```bibtex
@article{aouadi2026moroccanmedmcqa,
  title   = {Evaluating Large Language Models on Moroccan Medical Faculty Examinations: The MoroccanMedMCQA-FR Benchmark},
  author  = {Aouadi, Hamza and Nfaoui, El Habib and El Garouani, Said},
  journal = {IEEE Access},
  year    = {2026},
  note    = {To appear}
}
```

and the dataset: DOI [10.57967/hf/8867](https://doi.org/10.57967/hf/8867).

## License

- **Code** in this repository: [Apache-2.0](LICENSE).
- **Dataset** (hosted separately on Hugging Face): CC BY-NC 4.0.
