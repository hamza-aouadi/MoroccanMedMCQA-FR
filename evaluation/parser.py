#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer extraction for MoroccanMedMCQA-FR modele outputs.

Finds the answer whther a model puts it first or last
"""


import re 

LETTERS = ["a","b","c","d","e"]
_MARK = re.compile(r"(?i)(?:r[ée]ponse|answer|bonne\s+r[ée]ponse|la\s+r[ée]ponse\s+est|correct\s+answer)\s*:?\s*")
_ONLY = re.compile(r"^[\s]*([A-Ea-e](?:\s*[,;/&]?\s*(?:and|et)?\s*[A-Ea-e])*)[\s\.]*$")
_RUN  = re.compile(r"^[\s]*([A-Ea-e]{2,5})[\s\.\)]*$")

def _strip_think(raw):
    return re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", " ", raw) if raw else raw
    
    



def is_degenerate(raw):
    if not raw:
        return False
    t = raw.strip(); toks=t.split()
    if len(toks)>=20 and len(set(toks)) / max(1, len(toks))<0.15:
        return True
    if re.search(r"(?i)(\b\w[\w'?]{0,15}\b[\s,?.!]*)\1{7,}", t):
        return True
    return False
    

def _letters(s, present):
    ls = [c.lower() for c in re.findall(r"\b[A-Ea-e]\b", s or "") if c.lower() in present]
    if ls:
        return ls
    if s:
        m = _RUN.match(s.strip())
        if m:
            return [c.lower() for c in m.group(1) if c.lower() in present]
    return []



def _candidate_spans(raw):
    txt = _strip_think(raw) or ""
    if _MARK.search(txt):
        yield _MARK.split(txt)[-1]
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    only = [ln for ln in lines if _ONLY.match(ln)]
    for ln in reversed(only):
        yield ln
    yield txt




def parse_mcqu(raw, present):
    if is_degenerate(raw):
        return None
    for span in _candidate_spans(raw):
        ls = _letters(span, present)
        if ls:
            return ls[-1]
    return None





def parse_mcqm(raw, present):
    if is_degenerate(raw):
        return None
    txt = _strip_think(raw) or ""
    for span in _candidate_spans(raw):
        none_here = bool(re.search(r"(?i)\bnone\b", span))
        ls = set(_letters(re.sub(r"(?i)\bnone\b", " ", span), present))
        if ls:
            return ls
        if none_here:
            return set()
    if re.search(r"(?i)\bnone\b", txt):
        return set()
    return None

    


def _has_clean_answer(raw):
    raw = str(raw or "").strip()
    if len(raw) <=40:
        return True
    if re.search(r'(?i)(r[ée]ponse|answer|bonne r[ée]ponse)\s*[:\-]', raw):
        tail = re.split(r'(?i)(?:r[ée]ponse|answer|bonne r[ée]ponse)\s*[:\-]', raw)[-1]
        if re.search(r'[A-Ea-e]', tail):
            return True
    body = _strip_think(raw).strip()
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    pat = r'[A-Ea-e](\s*[,/]\s*[A-Ea-e])*'
    if lines and (re.fullmatch(pat, lines[-1]) or re.fullmatch(pat, lines[0])):
        return True
    return False






def parse_mcqu_strict(raw, present):
    body = _strip_think(raw or "").strip()
    first = next((l.strip() for l in body.splitlines() if l.strip()), "")
    m = re.match(r'(?i)^\(?([A-E])(?:[\.\)\:,\-\s]|$)', first)
    if m and m.group(1).lower() in present:
        return m.group(1).lower()
    if not _has_clean_answer(raw):
        return None
    return parse_mcqu(raw, present)



def parse_mcqm_strict(raw, present):
    if not _has_clean_answer(raw):
        return None
    return parse_mcqm(raw, present)