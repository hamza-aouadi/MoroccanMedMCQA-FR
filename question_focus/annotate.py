#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive-focus annotation (Understanding/Reasoning) for MoroccanMedMCQA-FR.

1. set your key: export OPENAI_API_KEY=sk-...
2. Edit Config Below.
3. Run: python annotate.py

Resume-safe: re-running continues where it stopped.
Use MODEL = gpt-4o for the main run, gpt-5.5 for the validation check.
"""

import os, json, time
from openai import OpenAI

# ======================= CONFIG ===================
DATA_PATH = "inputs/question_focus_validation_BLIND.xlsx"
OUT_PATH = "results/question_focus_labels.jsonl"
MODEL = "gpt-4o-2024-08-06"
EFFORT = "high"
# ===============================================

PROMPT = """You are an experienced medical doctor and independent practitioner. Your task is to determine whether a medical question primarily challenges the answerer's medical knowledge understanding or medical reasoning skills.
You will be given a medical exam question in French, optionally preceded by a clinical scenario. Please determine whether the question primarily challenges the answerer's medical knowledge understanding or medical reasoning ability. 
Reasoning: complicated, reasoning-heavy questions.
Understanding: little to no reasoning, instead assessing recall of medical knowledge.
Directly output either "Understanding" or "Reasoning" as your answer, without any additional information or explanation."""

client = OpenAI()

def clean(v):
    s = "" if v is None else str(v).strip()
    return None if s == "" or s.lower() == "nan" else s
    
    
def build_prompt(rec):
    parts = [PROMPT, ""]
    if rec["context"]:
        parts += ["**Clinical scenario:**", rec["context"]]
    parts += ["**Question:**", rec["question"], "**Output:**"]
    return "\n".join(parts)


def parse_label(raw):
    t = (raw or "").lower()
    if "reasoning" in t and "understanding" not in t:
        return "Reasoning"
    if "understanding" in t and "reasoning" not in t:
        return "Understanding"
    return None



def ask(prompt):
    if MODEL.startswith("gpt-5"):
        r = client.responses.create(
            model=MODEL, reasoning={"effort": EFFORT}, timeout=120,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}])
        return r.output_text
    r = client.chat.completions.create(
    model=MODEL, temperature=0, max_tokens=5, timeout=30,
    messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content
 


def load(path):
    if path.endswith(".json"):
        rows = json.load(open(path, encoding="utf-8"))
    else:
        import pandas as pd
        df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path, sep=";")
        rows = df.to_dict("records")
    return [{"id": int(float(r["id"])),
            "context": clean(r.get("context")),
            "question": clean(r.get("question"))}
            for r in rows if clean(r.get("id")) is not None]


done = set()
if os.path.exists(OUT_PATH):
            for line in open(OUT_PATH, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass

data = load(DATA_PATH)
todo = [r for r in data if r["id"] not in done]
print(f"total {len(data)} | already done {len(done)} | to label {len(todo)}")

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "a" , encoding="utf-8") as out:
    for i,rec in enumerate(todo, 1):
        raw = None
        for attempt in range(5):
            try:
                raw = ask(build_prompt(rec))
                break
            except Exception as e:
                print(f" retry {attempt+1}/5 id={rec['id']} ({type(e).__name__})") 
                time.sleep(2 ** (attempt + 1))
        label = parse_label(raw)
        out.write(json.dumps({"id": rec["id"], "question_focus": label, "raw": raw},
                             ensure_ascii=False) + "\n")
        out.flush()
        if i%100 == 0:
            print(f" {i}/{len(todo)}")

print("done ->", OUT_PATH)