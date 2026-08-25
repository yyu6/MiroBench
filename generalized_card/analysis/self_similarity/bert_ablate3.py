"""Split the URL effect: machine-generated media attachments vs human references."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_real_comments
from score_thread_self_bertscore import load_bert_scorer
from score_thread_self_bleu import tokenize
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]

URL=re.compile(r"https?://\S+|\bwww\.\S+")
MEDIA_HOSTS=("preview.redd.it","i.redd.it","i.imgur.com","imgur.com","v.redd.it","redd.it")
def is_media(u): return any(h in u.lower() for h in MEDIA_HOSTS)
def strip_media(t):  return re.sub(r"\s+"," ",URL.sub(lambda m:"" if is_media(m.group()) else m.group(), t)).strip()
def strip_ref(t):    return re.sub(r"\s+"," ",URL.sub(lambda m:"" if not is_media(m.group()) else m.group(), t)).strip()
def strip_all(t):    return re.sub(r"\s+"," ",URL.sub("",t)).strip()

cache={}; threads=[]
for p in pool[:50]:
    d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache: cache[d]=load_real_comments(d)[0]
    cs=cache[d].get(p["source_raw_post_id"]) or []
    if 8<=len(cs)<=26: threads.append((p["source_raw_post_id"],[c.text for c in cs]))
threads=threads[:20]

allr=[c for p in pool[:50] for c in
      ([x.text for x in (cache[REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]].get(p["source_raw_post_id"]) or [])])]
u=[x for t in allr for x in URL.findall(t)]
print(f"real comments {len(allr)}  URLs {len(u)}  media {sum(1 for x in u if is_media(x))}  reference {sum(1 for x in u if not is_media(x))}")
print(f"comment share with a media URL     {np.mean([1.0 if any(is_media(x) for x in URL.findall(t)) else 0.0 for t in allr]):.4f}")
print(f"comment share with a reference URL {np.mean([1.0 if any(not is_media(x) for x in URL.findall(t)) else 0.0 for t in allr]):.4f}")
mt=np.mean([len(tokenize(x)) for x in u if is_media(x)]); rt=np.mean([len(tokenize(x)) for x in u if not is_media(x)])
print(f"tokens per URL: media {mt:.1f}   reference {rt:.1f}")

scorer,_,dev,_,_,_=load_bert_scorer(bert_score_path=REPO/"bert_score-master",
    model_type="microsoft/deberta-xlarge-mnli", num_layers=None, batch_size=8,
    device="auto", idf=False, idf_sents=[], rescale_with_baseline=False, local_files_only=True)
def tf1(texts):
    c,r=[],[]
    for i in range(len(texts)):
        for j in range(i+1,len(texts)): c.append(texts[i]); r.append(texts[j])
    _,_,f1=scorer.score(c,r,batch_size=8); return float(np.mean([float(x) for x in f1]))
base=[tf1(t) for _,t in threads]
print(f"\n{'real (baseline)':<34}{np.mean(base):.4f}")
for name,fn in (("- machine media URLs only",strip_media),("- human reference URLs only",strip_ref),("- all URLs",strip_all)):
    v=[tf1([fn(x) for x in t]) for _,t in threads]; d=np.mean(v)-np.mean(base)
    print(f"{name:<34}{np.mean(v):.4f}   {d:+.4f}  = {100*d/0.0124:+.0f}% of the gap")
