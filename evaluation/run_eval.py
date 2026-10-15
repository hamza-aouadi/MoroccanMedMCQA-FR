#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a model (or all 36)over MoroccanMedMCQA-FR test set
Prompts are read from prompts/mcqu.txt and prompts/mcqm.txt

Keys:
    OpenRoutet  -> set OPENROUTER_API_KEY (or set it at prompt).
    Vertex      -> deploy the model, the npaste its endpoint URL+access token when the runners ask
    
"""

import os, sys, json
import parser as P
import openrouter, vertex

# ================= CONFIG ================
TEST_PATH = "inputs/MoroccanMedMCQA-FR_test_no_answers.json"
OUT_DIR = "results"
# =========================================

LETTERS = ["a","b", "c", "d", "e"]
HERE = os.path.dirname(os.path.abspath(__file__))
MCQU_TEMPLATE = open(os.path.join(HERE, "prompts", "mcqu.txt"), encoding="utf-8").read()
MCQM_TEMPLATE = open(os.path.join(HERE, "prompts", "mcqm.txt"), encoding="utf-8").read()


# ---- the 36 model---
MODELS = [  
    # ---------------- OpenRouter ----------------
    {"label": "gpt-5.5-high",            "provider": "openrouter", "model": "openai/gpt-5.5",               "reasoning": "high"},
    {"label": "gpt-5.5-none",            "provider": "openrouter", "model": "openai/gpt-5.5",               "reasoning": False, "temperature": 0},
    {"label": "gpt-4o",                  "provider": "openrouter", "model": "openai/gpt-4o-2024-08-06",     "temperature": 0},
    {"label": "claude-sonnet5-off",      "provider": "openrouter", "model": "anthropic/claude-sonnet-5",    "reasoning": False},
    {"label": "claude-sonnet5-high",     "provider": "openrouter", "model": "anthropic/claude-sonnet-5",    "reasoning": "high"},
    {"label": "deepseek-v4pro-off",      "provider": "openrouter", "model": "deepseek/deepseek-v4-pro",     "reasoning": False},
    {"label": "deepseek-v4pro-high",     "provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},     # default = reasoning ON
    {"label": "qwen3.8-27b-off",         "provider": "openrouter", "model": "qwen/qwen3.8-27b",             "reasoning": False},
    {"label": "mistral-medium3.5-none",  "provider": "openrouter", "model": "mistralai/mistral-medium-3-5", "temperature": 0},
    {"label": "mistral-medium3.5-high",  "provider": "openrouter", "model": "mistralai/mistral-medium-3-5", "reasoning": "high", "max_tokens": 8192},
    {"label": "llama4-scout",            "provider": "openrouter", "model": "meta-llama/llama-4-scout",     "temperature": 0},
    {"label": "llama4-maverick",         "provider": "openrouter", "model": "meta-llama/llama-4-maverick",  "temperature": 0},
    {"label": "kimi-k3-low",             "provider": "openrouter", "model": "moonshotai/kimi-k3",           "reasoning": "low", "max_tokens": 8192},
    {"label": "glm-5.2-high",            "provider": "openrouter", "model": "z-ai/glm-5.2",                 "reasoning": "high", "temperature": 0, "max_tokens": 8192},
    {"label": "gemini-3.1-pro-low",      "provider": "openrouter", "model": "google/gemini-3.1-pro-preview",     "reasoning": "low",     "temperature": 0},
    {"label": "gemini-3.1-pro-high",     "provider": "openrouter", "model": "google/gemini-3.1-pro-preview",     "reasoning": "high",    "temperature": 0},
    {"label": "gemini-3.5-flash-minimal","provider": "openrouter", "model": "google/gemini-3.5-flash",           "reasoning": "minimal", "temperature": 0},
    {"label": "gemini-3.5-flash-high",   "provider": "openrouter", "model": "google/gemini-3.5-flash",           "reasoning": "high",    "temperature": 0},
    {"label": "gemini-3.5-flash-lite-minimal","provider": "openrouter", "model": "google/gemini-3.5-flash-lite", "reasoning": "minimal", "temperature": 0},
    {"label": "gemini-3.5-flash-lite-high",   "provider": "openrouter", "model": "google/gemini-3.5-flash-lite", "reasoning": "high",    "temperature": 0},

    # ---------------- Vertex ----------------
    {"label": "grok-4.20-nonreasoning",  "provider": "vertex", "model": "xai/grok-4.20-non-reasoning", "temperature": 0, "max_tokens": 2048},
    {"label": "grok-4.20-reasoning-high","provider": "vertex", "model": "xai/grok-4.20-reasoning",     "reasoning_effort": "high", "max_tokens": 8192},
    {"label": "gpt-oss-120b-low",        "provider": "vertex", "model": "openai/gpt-oss-120b", "extra_body": {"reasoning_effort": "low"},  "temperature": 0, "max_tokens": 8192},
    {"label": "gpt-oss-120b-high",       "provider": "vertex", "model": "openai/gpt-oss-120b", "extra_body": {"reasoning_effort": "high"}, "temperature": 0, "max_tokens": 8192},
    {"label": "gpt-oss-20b-low",         "provider": "vertex", "model": "openai/gpt-oss-20b",  "extra_body": {"reasoning_effort": "low"},  "temperature": 0, "max_tokens": 8192},
    {"label": "gpt-oss-20b-high",        "provider": "vertex", "model": "openai/gpt-oss-20b",  "extra_body": {"reasoning_effort": "high"}, "temperature": 0, "max_tokens": 8192},
    {"label": "gemma-4-31b-it-on",       "provider": "vertex", "model": "google/gemma-4-31b-it", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},  "temperature": 0, "max_tokens": 8192},
    {"label": "gemma-4-31b-it-off",      "provider": "vertex", "model": "google/gemma-4-31b-it", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "temperature": 0, "max_tokens": 2048},
    {"label": "gemma-4-12b-it-off",      "provider": "vertex", "model": "google/gemma-4-12b-it", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "temperature": 0, "max_tokens": 2048},
    {"label": "phi-4",                   "provider": "vertex", "model": "microsoft/phi-4",            "temperature": 0, "max_tokens": 2048, "parser": "strict"},
    {"label": "medgemma-4b-it",          "provider": "vertex", "model": "google/medgemma-4b-it",      "temperature": 0, "max_tokens": 256},
    {"label": "medgemma-27b-it",         "provider": "vertex", "model": "google/medgemma-27b-it",     "temperature": 0, "max_tokens": 256},
    {"label": "medgemma-1.5-4b-it",      "provider": "vertex", "model": "google/medgemma-1.5-4b-it",  "temperature": 0, "max_tokens": 2048},
    {"label": "biomistral-7b",           "provider": "vertex", "model": "BioMistral/BioMistral-7B",         "temperature": 0, "max_tokens": 1024, "parser": "strict"},
    {"label": "openbiollm-8b",           "provider": "vertex", "model": "aaditya/Llama3-OpenBioLLM-8B",     "temperature": 0, "max_tokens": 1024, "parser": "strict"},
    {"label": "openbiollm-70b",          "provider": "vertex", "model": "aaditya/Llama3-OpenBioLLM-70B",    "temperature": 0, "max_tokens": 512,  "parser": "strict"},
]




def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("data", d) if isinstance(d, dict) else d
    



def options(rec):
    a = rec.get("answers", {})
    return {L: a[L] for L in LETTERS if a.get(L) not in (None, "") and str(a.get(L)).strip()}



def render(template, rec):
    o = options(rec)
    c = rec.get("context")
    scenario = f"**Clinical scenario:**\n{str(c).strip()}\n" if c and str(c).strip() else ""
    opts = "\n".join(f"{L.upper()}. {o[L]}" for L in LETTERS if L in o)
    return template.format(scenario=scenario, question=rec["question"],options=opts)
 
 
 
 
 


def predict(cfg, qt, raw, present):
    strict = cfg.get("parser") == "strict"
    if qt == "MCQU":
        p = P.parse_mcqu_strict(raw, present) if strict else P.parse_mcqu(raw, present)
        return p or "UNPARSEABLE"
    s = P.parse_mcqm_strict(raw, present) if strict else P.parse_mcqm(raw, present)
    return "UNPARSEABLE" if s is None else "|".join(sorted(s))
    
    



def run_model(cfg, records):
    call = openrouter.call if cfg["provider"] == "openrouter" else vertex.call
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{cfg['label']}_predictions.jsonl")
    with open (out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            qt = rec.get("question_type")
            template = MCQU_TEMPLATE if qt == "MCQU" else MCQM_TEMPLATE
            try:
                raw = call(cfg, render(template, rec))
                pred = predict(cfg, qt, raw, set(options(rec)))
            except Exception as e:
                raw, pred = f"ERROR: {type(e).__name__}:{e}", "ERROR"
            f.write(json.dumps({"id": rec.get("id"), "question_type": qt, "question_focus": rec.get("question_focus"), "speciality": rec.get("speciality"), "pred": pred, "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if i%50 ==0:
                print(f" {i}/{len(records)}")
    print(f"    saved {len(records)} -> {out_path}")




def menu():
    print("\nMoroccanMedMCQA-FR - model runner\n")
    for i, m in enumerate(MODELS, 1):
        print(f"    {i:2d}. {m['label']:28s} [{m['provider']}]")
    print(" 0. All models\n")
    choice = input("Pick a number (0 for all): ").strip()
    if choice == "0":
        return MODELS
    return [MODELS[int(choice) -1]]



if __name__ == "__main__":
    records = load(TEST_PATH)
    print(f"loaded {len(records)} test questions from {TEST_PATH}")
    for cfg in menu():
        print(f"\n=== {cfg['label']} ({cfg['provider']}) ===")
        run_model(cfg, records)
    print("\nDONE.")
    
