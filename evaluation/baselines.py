"""
Baseline Predictions for MoroccanMedMCQA-FR.

Generates the naive-baseline predictions for every test question.

  MCQU : Random-1, Always-A, Longest, Majority baselines   
  MCQM : Random-k, All-options baselines                   -> set of letters ("a|c")


Edit CONFIG below, then run:  python baselines.py
"""

import os, csv, json, random

# CONFIGURATION
TEST_PATH       = "inputs/MoroccanMedMCQA-FR_test_no_answers.json"   
OUT_MCQU        = "results/mcqu_baseline_predictions.csv"
OUT_MCQM        = "results/mcqm_baseline_predictions.csv"
SEED            = 42
RANDOM_P        = 0.5       # probability for each Random-k
MAJORITY_LETTER = "b"      # most frequent answer opiotn in dataset


LETTERS = ["a", "b", "c", "d", "e"]


def load(path, qtype):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", next(iter(data.values())))
    return [r for r in data if r.get("question_type") == qtype]


def options(rec):
    """Return the availablle answer choices for a question"""
    ans = rec.get("answers", {})
    return {L: str(ans[L]).strip() for L in LETTERS
            if ans.get(L) not in (None, "") and str(ans.get(L)).strip()}


# generate predictions
def mcqu_row():
    rows = []
    for r in load(TEST_PATH, "MCQU"):
        o = options(r)
        rng = random.Random(f"{SEED}-{r.get('id')}")           
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
        available_options = sorted(o)
        rng = random.Random(f"{SEED}-{r.get('id')}")           
        random_k = [L for L in available_options if rng.random() < RANDOM_P]
        rows.append({
            "id":               r.get("id"),
            "speciality":       r.get("speciality"),
            "question_focus":   r.get("question_focus"),
            "num_options":      len(o),
            "pred_random_k":    "|".join(random_k),
            "pred_all_options": "|".join(available_options),
        })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    u, m = mcqu_row(), mcqm_rows()
    write_csv(OUT_MCQU, u)
    write_csv(OUT_MCQM, m)
    print(f"MCQU predictions : {len(u)} -> {OUT_MCQU}")
    print(f"MCQM predictions : {len(m)} -> {OUT_MCQM}")
