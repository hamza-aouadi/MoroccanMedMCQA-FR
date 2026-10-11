#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline PREDICTIONS for MoroccanMedMCQA-FR.

Generates the naive-baseline predictions for every test question.

  MCQU : Random-1, Always-A, Longest, Majority   -> one letter
  MCQM : Random-k, All-options                   -> set of letters ("a|c")


Edit CONFIG below, then run:  python baselines.py
"""

import os, csv, json, random

# ============== CONFIG ==============
TEST_PATH       = "inputs/MoroccanMedMCQA-FR_test_no_answers.json"   
OUT_MCQU        = "results/mcqu_baseline_predictions.csv"
OUT_MCQM        = "results/mcqm_baseline_predictions.csv"
SEED            = 42        # per-question seed (reproducible Random-1 / Random-k)
RANDOM_P        = 0.5       # per-option inclusion probability for Random-k
MAJORITY_LETTER = "b"       # most-frequent correct letter
# ====================================

LETTERS = ["a", "b", "c", "d", "e"]


def load(path, qtype):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", next(iter(data.values())))
    return [r for r in data if r.get("question_type") == qtype]


def options(rec):
    """{letter: text} for options that are actually present (non-empty)."""
    ans = rec.get("answers", {})
    return {L: str(ans[L]).strip() for L in LETTERS
            if ans.get(L) not in (None, "") and str(ans.get(L)).strip()}


# =========================== predictions ===========================
def mcqu_rows():
    rows = []
    for r in load(TEST_PATH, "MCQU"):
        o = options(r)
        rng = random.Random(f"{SEED}-{r.get('id')}")           # per-question seed
        rows.append({
            "id":             r.get("id"),
            "speciality":     r.get("speciality"),
            "question_focus": r.get("question_focus"),
            "num_options":    len(o),
            "pred_random1":   rng.choice(list(o)) if o else "",
            "pred_always_a":  "a",
            "pred_longest":   max(o, key=lambda L: (len(o[L]), -LETTERS.index(L))) if o else "",
            "pred_majority":  MAJORITY_LETTER,
        })
    return rows


def mcqm_rows():
    rows = []
    for r in load(TEST_PATH, "MCQM"):
        o = options(r)
        present = sorted(o)
        rng = random.Random(f"{SEED}-{r.get('id')}")           # per-question seed
        random_k = [L for L in present if rng.random() < RANDOM_P]
        rows.append({
            "id":               r.get("id"),
            "speciality":       r.get("speciality"),
            "question_focus":   r.get("question_focus"),
            "num_options":      len(o),
            "pred_random_k":    "|".join(random_k),
            "pred_all_options": "|".join(present),
        })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    u, m = mcqu_rows(), mcqm_rows()
    write_csv(OUT_MCQU, u)
    write_csv(OUT_MCQM, m)
    print(f"MCQU predictions : {len(u)} -> {OUT_MCQU}")
    print(f"MCQM predictions : {len(m)} -> {OUT_MCQM}")
