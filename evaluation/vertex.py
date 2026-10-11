#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertex AI call function
Deploy the model manually, then give the runner its endpoint: base url and token
"""

import getpass
from openai import OpenAI

ENDPOINTS = {}


def set_endpoint(label, base_url, token):
    ENDPOINTS[label] = {"base_url": base_url, "token": token, "client": OpenAI(base_url=base_url, api_key=token)}




def _client(cfg):
    e = ENDPOINTS.get(cfg["label"])
    if e is None:
        print(f" Vertex model '{cfg['label']}' - deploy it, then pasteits endpoint below.")
        base = input(" endpoint base URL: ").strip()
        token = getpass.getpass("   access token: ").strip()
        set_endpoint(cfg["label"], base, token)
        e = ENDPOINTS[cfg["label"]]
    return e["client"]




def call(cfg, prompt):
    client = _client(cfg)
    kw = {"model": cfg["model"], "messages": _messages(cfg, prompt)}
    if cfg.get("reasoning_effort"):
        kw["reasoning_effort"] = cfg["reasoning_effort"]
    if cfg.get("extra_body"):
        kw["extra_body"] = cfg["extra_body"] 
    if cfg.get("temperature") is not None:
        kw["temperature"] = cfg["temperature"] 
    if cfg.get("max_tokens") is not None:
        kw["max_tokens"] = cfg["max_tokens"] 
    r = client.chat.completions.create(**kw)
    return r.choices[0].message.content or ""




def _messages(cfg, prompt):
    msgs = []
    if cfg.get("system"):
        msgs.append({"role": "system", "content": cfg["system"]})
    msgs.append({"role":"user", "content": prompt})
    return msgs