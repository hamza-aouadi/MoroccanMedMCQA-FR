#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score model predictions for MoroccanMedMCQA-FR
Joins each predictions file with gold answers:
    MCQU Accuracy | MCQM EMR | MCQM Hamming
"""



import os, sys, re, json, glob

LETTERS = ["a", "b", "c", "d", "e"]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("data", d) if isinstance(d, dict) else d
    
    


def gold_map(path):
    g = {}
    for r in load_json(path):
        ca = r.get("correct_answers", [])
        if isinstance(ca, str):
            ca = re.split(r"[^a-eA-E]+", ca)
        gold = {c.lower() for c in ca if c and str(c).strip().lower() in LETTERS}
        g[r.get("id")] = {"qt": r.get("question_type"), "gold": gold}
    return g



def to_set(pred):
    p = (pred or "").strip()
    if p in ("UNPARSEABLE", "ERROR"):
        return None
    return {x for x in p.split("|") if x}
    
def jaccard (g, p):
    u = g | p
    return 1.0 if not u else len(g & p) / len(u)
    


def tr(x):
    return int(x * 100) / 100 if x ==x else x


def score(pred_path, gold):
    mcqu, emr, ham = [], [], []
    for line in open(pred_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        info = gold.get(row.get("id"))
        if info is None:
            continue
        qt, pred = row.get("question_type"), row.get("pred")
        if qt == "MCQU":
            if len(info["gold"]) != 1:
                continue
            mcqu.append(int(pred == next(iter(info["gold"]))))
        elif qt == "MCQM":
            G, P = info["gold"], to_set(pred)
            if P is None:
                emr.append(0); ham.append(0.0)
            else:
                emr.append(int(P == G)); ham.append(jaccard(G, P))
    pct = lambda xs: tr(100 * sum(xs) / len(xs)) if xs else float("nan")
    return pct(mcqu), pct(emr), pct(ham)





def main():
    files = [a for a in sys.argv[1:]] or sorted(glob.glob("results/*_predictions.jsonl"))
    if not files:
        sys.exit("no predictions files found(results/*_predictions.jsonl)")
    gold_path = input("Path to answered test JSON").strip().strip('""')
    gold = gold_map(gold_path)

    print(f"\n{'model':34s}{'MCQU Acc':>10}{'MCQM EMR':>10}{'Hamming':>10}")
    print("-" * 64)
    for fp in files:
        name = os.path.basename(fp).replace("_predictions.jsonl", "")
        acc, e, h = score(fp, gold)
        print(f"{name:34s}{acc:>10.2f}{e:>10.2f}{h:>10.2f}")
        

if __name__ == "__main__":
    main()