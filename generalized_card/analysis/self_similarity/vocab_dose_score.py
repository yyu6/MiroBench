#!/usr/bin/env python3
"""Score the constructed threads with the SAME BERTScore config as the eval
suite (deberta-xlarge-mnli, L40, no idf, no baseline rescale, cpu), plus the
semantic-cosine control (all-mpnet-base-v2)."""
import json, sys, os, itertools, statistics
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "bert_score-master"))
IN = Path("/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/d8816651-1679-43a5-8d4b-21a1a35e5936/scratchpad/vocab_dose_threads.json")
OUT = IN.with_name("vocab_dose_scored.json")

specs = json.loads(IN.read_text())
import torch
from bert_score import BERTScorer
scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40,
                    batch_size=32, idf=False, device="cpu", lang="en",
                    rescale_with_baseline=False)
print("bert hash:", scorer.hash, flush=True)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

rows = []
for n, sp in enumerate(specs):
    texts = sp["texts"]
    pairs = list(itertools.combinations(range(len(texts)), 2))
    cands = [texts[i] for i, _ in pairs]
    refs  = [texts[j] for _, j in pairs]
    P, R, F = scorer.score(cands, refs, batch_size=64)
    f1 = F.tolist()
    emb = st.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    sim = (emb @ emb.T).tolist()
    cos = [sim[i][j] for i, j in pairs]
    row = {**{k: v for k, v in sp.items() if k != "texts"},
           "bert_mean_f1": statistics.mean(f1),
           "bert_median_f1": statistics.median(f1),
           "cos_mean": statistics.mean(cos)}
    rows.append(row)
    print(f"[{n+1}/{len(specs)}] types={row['types']} tok={row['tokens']} "
          f"bertF1={row['bert_mean_f1']:.4f} cos={row['cos_mean']:.4f}", flush=True)
OUT.write_text(json.dumps(rows, indent=1))
print("wrote", OUT)
