"""What a real multi-link comment looks like, and what v117 got wrong about it.

G61 / FINDINGS s14: v117 hits its metric target exactly (1.68 URLs per carrying
comment against real's 1.67) and stacks four unrelated links at the end of a
46-word comment. The links are drawn from an 802-entry inventory by hash with no
relation to each other or to the comment.

This measures the two structural facts a v118 needs, on the evaluation-excluded
corpus only (no seed thread is read), and prints the inventory-side feasibility
of honouring them:

1. HOST COHERENCE -- given k URLs in one comment, how often do they share a host,
   conditional on k. The pooled 0.643 quoted in s14 is over all k>=2 and hides
   the k-dependence, which is what an implementation actually needs.
2. PLACEMENT -- where the first URL sits as a fraction of the comment, measured
   in words rather than characters so it composes with a word-count cue.

`youtu.be` and `youtube.com` are the same source to a reader and different
netlocs to `urlparse`, so both the raw netloc and a folded registrable-domain
form are reported. The folded form is the one an implementation should use.

Usage:  python3 generalized_card/analysis/self_similarity/url_host_coherence.py
"""
from __future__ import annotations

import collections
import statistics as st
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "generalized_card"))

from generalized_card.reference_link import (  # noqa: E402
    MAX_LINKS_PER_SLOT,
    MAX_URL_CHARS,
    extract_urls,
    is_media_url,
)
from generalized_card.data import load_real_thread_bank  # noqa: E402

SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
RAW = REPO / "data/raw/discussions/camera_product"

# Same source, different netloc. Folded before counting distinct hosts, because a
# reader sees one place and `urlparse` sees two.
_FOLD = {
    "youtu.be": "youtube.com",
    "m.youtube.com": "youtube.com",
    "np.reddit.com": "reddit.com",
    "old.reddit.com": "reddit.com",
    "new.reddit.com": "reddit.com",
    "redd.it": "reddit.com",
    "a.co": "amazon.com",
    "amzn.to": "amazon.com",
}


def host_of(url: str, *, fold: bool = True) -> str:
    raw = url if url.lower().startswith("http") else "http://" + url
    netloc = urlparse(raw).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if not fold:
        return netloc
    if netloc in _FOLD:
        return _FOLD[netloc]
    parts = netloc.split(".")
    # en.wikipedia.org -> wikipedia.org; usa.canon.com -> canon.com. Two labels is
    # wrong for co.uk, which this corpus has too few of to matter.
    return ".".join(parts[-2:]) if len(parts) > 2 else netloc


def excluded_threads() -> list[dict]:
    import json

    excluded = {
        str(row.get("source_raw_post_id") or "").strip()
        for row in json.load(open(SEED_POOL)).get("seed_posts") or []
    }
    unique: dict[str, dict] = {}
    for thread in load_real_thread_bank(RAW):
        post_id = str(thread.get("post_id") or "").strip()
        if not post_id or post_id in excluded:
            continue
        old = unique.get(post_id)
        if old is None or int(thread.get("comment_count") or 0) > int(old.get("comment_count") or 0):
            unique[post_id] = thread
    return sorted(unique.values(), key=lambda row: str(row.get("post_id") or ""))


def main() -> None:
    threads = excluded_threads()
    comments = 0
    carriers: list[tuple[str, list[str]]] = []
    for thread in threads:
        for comment in thread.get("comments") or []:
            body = str(comment.get("body") or "")
            if not body.strip():
                continue
            comments += 1
            urls = [u for u in extract_urls(body) if not is_media_url(u)]
            urls = [u for u in urls if 8 <= len(u) <= MAX_URL_CHARS]
            if urls:
                carriers.append((body, urls))

    print(f"threads {len(threads)}  comments {comments}  carriers {len(carriers)}")
    multi = [(b, u) for b, u in carriers if len(u) >= 2]
    print(f"carriers with 2+ URLs: {len(multi)}\n")

    # ---- 1. host coherence, conditional on k --------------------------------
    print("== distinct hosts given k URLs (folded / raw netloc) ==")
    print(f"{'k':>3}{'n':>6}{'all one host':>14}{'raw netloc':>12}{'distinct hosts':>28}")
    pooled_same = pooled_n = 0
    for k in range(2, 9):
        rows = [u for _, u in multi if len(u) == k]
        if not rows:
            continue
        folded = [len({host_of(x) for x in u}) for u in rows]
        raw = [len({host_of(x, fold=False) for x in u}) for u in rows]
        same = sum(1 for d in folded if d == 1)
        dist = collections.Counter(folded)
        if k <= MAX_LINKS_PER_SLOT:
            pooled_same += same
            pooled_n += len(rows)
        print(
            f"{k:>3}{len(rows):>6}{same / len(rows):>13.3f}"
            f"{sum(1 for d in raw if d == 1) / len(rows):>12.3f}"
            f"{str(dict(sorted(dist.items()))):>28}"
        )
    allrows = [u for _, u in multi]
    folded_all = [len({host_of(x) for x in u}) for u in allrows]
    print(
        f"\npooled k>=2 : all-one-host {sum(1 for d in folded_all if d == 1) / len(allrows):.3f}"
        f"  ({sum(1 for d in folded_all if d == 1)}/{len(allrows)})"
        f"   distinct-host dist {dict(sorted(collections.Counter(folded_all).items()))}"
    )
    print(
        f"pooled 2<=k<={MAX_LINKS_PER_SLOT}: all-one-host {pooled_same / pooled_n:.3f}"
        f"  ({pooled_same}/{pooled_n})  <- the rate an arm capped at "
        f"{MAX_LINKS_PER_SLOT} should draw"
    )

    # ---- 2. placement of the first URL --------------------------------------
    print("\n== where the URLs sit, in words ==")

    def positions(body: str, urls: list[str]) -> list[float]:
        words = body.split()
        if len(words) <= 1:
            return []
        out = []
        for url in urls:
            hit = next((i for i, w in enumerate(words) if url in w), None)
            if hit is not None:
                out.append(hit / (len(words) - 1))
        return out

    for label, rows in (("all carriers", carriers), (f"2<=k<={MAX_LINKS_PER_SLOT}", [
        (b, u) for b, u in multi if len(u) <= MAX_LINKS_PER_SLOT
    ])):
        firsts, lasts, words = [], [], []
        trailing = 0
        for body, urls in rows:
            pos = positions(body, urls)
            if not pos:
                continue
            firsts.append(min(pos))
            lasts.append(max(pos))
            words.append(len(body.split()))
            if min(pos) >= 0.75:
                trailing += 1
        if not firsts:
            continue
        print(
            f"{label:<16} n={len(firsts):<5} first URL: median {st.median(firsts):.3f}"
            f"  mean {st.mean(firsts):.3f}   last URL median {st.median(lasts):.3f}"
        )
        print(
            f"{'':<16} share whose FIRST url is in the last quarter: {trailing / len(firsts):.3f}"
            f"   median comment words {st.median(words):.0f}"
        )
        quart = collections.Counter(min(3, int(f * 4)) for f in firsts)
        print(
            f"{'':<16} first-URL quartile "
            + "  ".join(f"q{i + 1} {quart.get(i, 0) / len(firsts):.3f}" for i in range(4))
        )

    # ---- 3. can the inventory honour a same-host draw? ----------------------
    print("\n== inventory feasibility ==")
    import json

    profile = REPO / "artifacts/generalized_card/runs/v117_calibration_20260826_v1/domain_profile.json"
    if profile.exists():
        inv = json.load(open(profile))["reference_link_inventory"]
        groups = collections.Counter(host_of(u) for u in inv["urls"])
        total = sum(groups.values())
        print(f"inventory urls {total}  folded hosts {len(groups)}")
        for k in range(2, MAX_LINKS_PER_SLOT + 1):
            usable = sum(c for c in groups.values() if c >= k)
            print(
                f"  urls inside a host group of size >= {k}: {usable:>4} "
                f"({usable / total:.3f})  groups {sum(1 for c in groups.values() if c >= k)}"
            )
    else:
        print(f"(no profile at {profile})")


if __name__ == "__main__":
    main()
