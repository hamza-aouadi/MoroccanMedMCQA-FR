#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Openrouter call function
Reads OPENROUTER_API_KEY from environment, or prompt once
"""

import os
import getpass
from openai import OpenAI

_client= None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OPENROUTER_API_KEY: ").strip()
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client



def _reasoning_block(cfg):
    r = cfg["reasoning"]
    if r is False:
        return {"enabled": False}
    if r is True:
        return {"enabled": True}
    if isinstance (r, int):
        return {"enabled": True, "max_tokens": r}
    return {"enabled": True, "effort": r}
    



def call(cfg, prompt):
    client = _get_client()
    kw = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
    if "reasoning" in cfg:
        kw["extra_body"] = {"reasoning": _reasoning_block(cfg)}
    if cfg.get("temperature") is not None:
        kw["temperature"] = cfg["temperature"]
    if cfg.get("max_tokens"):
        kw["max_tokens"] = cfg["max_tokens"]
    r  =client.chat.completions.create(**kw)
    return r.choices[0].message.content or ""