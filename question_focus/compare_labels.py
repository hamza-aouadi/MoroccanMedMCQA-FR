#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agreement + Cohen's Kappa between two cognitive-focus label files.

Edit config below, then run: python compare_labels.py
"""

import os, json
from collections import Counter

# ======================= CONFIG =============
FILE_A = "inputs/question_focus_validation_BLIND_expert_1_classification.xlsx"
FILE_B = "inputs/question_focus_validation_BLIND_expert_2_classification.xlsx"
COL = "expert_label"
# ==============================================




def norm(v):
    s = str(v).strip().lower()
    if s.startswith("reason"):
        return "Reasoning"
    if s.startswith("under"):
        return "Understanding"
    return None






def load(path):
    labels = {}
    if path.endswith((".jsonl", ".json")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                o = json.loads(line)
                labels[int(o["id"])] = norm(o.get("question_focus") or o.get("label"))
    else:
        import pandas as pd
        df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
        df = df.dropna(subset=["id"])
        for _, r in df.iterrows():
            labels[int(float(r["id"]))] = norm(r.get(COL))
    return labels
    

A, B = load(FILE_A), load(FILE_B)
ids = [i for i in A if i in B and A[i] and B[i]]
a = [A[i] for i in ids]
b = [B[i] for i in ids]
n = len(ids)

po = sum(1 for x, y in zip(a, b) if x == y) / n
ca, cb = Counter(a), Counter(b)
pe = sum((ca[c] / n) * (cb[c] / n) for c in set(a) | set(b))
kappa = (po - pe) / (1 - pe)

print(f"compared        : {n} questions")
print(f"agreement       : {100*po:.2f}%")
print(f"Cohen's kappa   : {kappa:.4f}")
print(f"A distribution  : {dict(ca)}")
print(f"B distribution  : {dict(cb)}")