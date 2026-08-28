"""Which decoding knobs does gpt-5.4-mini actually accept? Never tested in this repo."""
import os, json, sys
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
for line in (REPO/"third_party/MiroFish/.env").read_text().splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
from openai import OpenAI
client=OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"))
MODEL="gpt-5.4-mini"
PROMPT="Write exactly one short Reddit comment about whether a compact camera is worth it for travel. Output only the comment body."
TRIALS=[
 ("baseline",                {}),
 ("temperature=1.3",         {"temperature":1.3}),
 ("temperature=0.7",         {"temperature":0.7}),
 ("top_p=0.85",              {"top_p":0.85}),
 ("frequency_penalty=0.8",   {"frequency_penalty":0.8}),
 ("presence_penalty=0.8",    {"presence_penalty":0.8}),
 ("reasoning_effort=minimal",{"reasoning_effort":"minimal"}),
 ("reasoning_effort=low",    {"reasoning_effort":"low"}),
 ("verbosity=low",           {"verbosity":"low"}),
]
for label,extra in TRIALS:
    kw={"model":MODEL,"messages":[{"role":"user","content":PROMPT}],"max_completion_tokens":400}
    kw.update(extra)
    try:
        r=client.chat.completions.create(**kw)
        txt=str(r.choices[0].message.content or "").strip()
        u=r.usage
        print(f"  OK   {label:<26} out_tok={u.completion_tokens:4d}  | {txt[:95]}")
    except Exception as e:
        msg=str(e).split('\n')[0][:130]
        print(f"  FAIL {label:<26} {type(e).__name__}: {msg}")
