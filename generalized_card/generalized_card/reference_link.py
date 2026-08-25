"""Drawn reference links for anonymous slots whose matched comment carried one.

Measured on the evaluation-excluded corpus (542 threads, 15,294 comments, zero
seed overlap): **4.40%** of real comments carry a human reference URL --
dpreview, youtube, flickr, camerasize, bhphotovideo, mpb -- and **0.00%** of
generated comments have ever carried one. Removing URLs from real text and
rescoring with the shipped BERTScore model moves real's `self_bertscore` by
+0.0094, which is **76%** of the whole generated-vs-real gap; the human
reference subset alone is +0.0064, or 52%. Nothing else tested moves it:
comma-joining -1%, u//r/ mentions 0%, emphasis characters -1%, quote markers
and escapes +1%. See `analysis/self_similarity/FINDINGS.md` s3.

Why the Writer has never written one, and what changes here:

- The routing already exists and is already right. `surface_contract.
  infer_surface_texture` tests for a URL **first**, so `surface_texture ==
  "link_reference"` holds exactly when the matched real comment carried a URL.
  On the N=50 artifact that is 56 slots, and the union with `evidence_mode ==
  "link_quote_reference"` is 70 of 1974 = 3.55% against the excluded corpus's
  4.40%.
- Three separate rules then forbid the Writer from producing one, and they are
  correct as written: a Writer with no source must not invent a URL. This module
  hands it a **real** URL from the evaluation-excluded corpus instead, so the
  prohibition on *inventing* stays in force under both arm values.

The matched comment's own URL is never used. Copying it would put evaluation-set
text verbatim into generated output, which `ORIENTATION.md` s4 forbids and
`output_audit`'s `matched_real_copy_risks` checks for -- and it would buy
nothing, because the metric effect comes from a high-entropy URL token being
present at all, not from which URL it is.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


REFERENCE_LINK_MODE = "off"

# Platform-generated image attachments, not authored text. They are excluded
# from the inventory: a text generator has no reason to emit the hash-bearing
# URL Reddit mints when a user attaches a photo. Measured separately, they carry
# 22% of the URL effect against the human references' 52%.
MEDIA_HOSTS = (
    "preview.redd.it",
    "i.redd.it",
    "v.redd.it",
    "i.imgur.com",
    "imgur.com",
)
# `\S+` was wrong and the v113 gate proved it: Reddit renders a bare link as
# `[url](url)` with `_` escaped to `\_`, and `\S+` runs straight through `](`
# into the second copy. 166 of 690 inventory entries (23.8%) were malformed that
# way, and 6 of the 23 links the gate wrote came out as `url](url` -- visible
# garbage in the output, not an invented URL. Stop at every bracket, paren,
# quote and backslash, and unescape Reddit's markdown escapes first.
_MD_ESCAPE_RE = re.compile(r"\\(?=[_*~\[\]()])")
URL_RE = re.compile(
    r"https?://[^\s<>\[\]()\"'\\]+|\bwww\.[^\s<>\[\]()\"'\\]+", re.I
)
_TRAILING = ".,;:!?*\u2019\"'"


def extract_urls(text: str) -> list[str]:
    """Pull clean URLs out of Reddit markdown, in order, with duplicates kept."""

    cleaned = _MD_ESCAPE_RE.sub("", str(text or ""))
    return [
        found
        for match in URL_RE.finditer(cleaned)
        if (found := match.group(0).rstrip(_TRAILING))
    ]
# Measured on the excluded corpus: median 60 characters, p75 99, p90 153, max
# 641. The cap keeps 97% of mentions and drops only the pathological tail, which
# would dominate a short slot's token count and inflate its own neighbours'
# self-BLEU floor.
MAX_URL_CHARS = 240
ROUTED_TEXTURES = ("link_reference",)
ROUTED_EVIDENCE_MODES = ("link_quote_reference",)


def set_reference_link_mode(mode: str) -> str:
    """Select whether routed slots are offered a drawn reference link."""

    global REFERENCE_LINK_MODE
    chosen = str(mode or "off").strip().lower()
    REFERENCE_LINK_MODE = "measured" if chosen == "measured" else "off"
    return REFERENCE_LINK_MODE


def reference_link_enabled() -> bool:
    return REFERENCE_LINK_MODE == "measured"


def is_media_url(url: str) -> bool:
    lowered = str(url or "").lower()
    return any(host in lowered for host in MEDIA_HOSTS)


def build_reference_link_inventory(
    reference_threads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect human reference URLs from evaluation-excluded threads only.

    The caller is responsible for the exclusion; this never reads a seed thread
    because it never sees one.
    """

    mentions: list[str] = []
    comment_count = 0
    carrying = 0
    for thread in reference_threads or []:
        for comment in thread.get("comments") or []:
            body = str(comment.get("body") or "")
            if not body.strip():
                continue
            comment_count += 1
            found = [url for url in extract_urls(body) if not is_media_url(url)]
            found = [url for url in found if 8 <= len(url) <= MAX_URL_CHARS]
            if found:
                carrying += 1
            mentions.extend(found)
    unique = sorted(set(mentions))
    return {
        "available": bool(unique),
        "urls": unique,
        "url_count": len(unique),
        "mention_count": len(mentions),
        "reference_comment_count": comment_count,
        "carrying_comment_share": (carrying / comment_count) if comment_count else 0.0,
        "max_url_chars": MAX_URL_CHARS,
        "excludes_media_hosts": list(MEDIA_HOSTS),
        "source": "evaluation-excluded threads only; no seed thread is read",
    }


def reference_link_slot(task: Any) -> bool:
    """Return whether this slot's matched comment carried a reference URL.

    Both fields are structural labels already assigned upstream. Neither exposes
    the matched comment's text.
    """

    texture = str(getattr(task, "surface_texture", "") or "")
    evidence = str(getattr(task, "evidence_mode", "") or "")
    return texture in ROUTED_TEXTURES or evidence in ROUTED_EVIDENCE_MODES


def draw_reference_link(task: Any, inventory: dict[str, Any] | None) -> str:
    """Draw one URL for this slot, deterministically and without replacement bias.

    Keyed on the slot's own identity so a rerun of the same slot draws the same
    link and two slots in one thread almost never collide -- 70 routed slots
    against 682 inventory entries. Repeated links would be repeated n-grams and
    would push `self_bleu_4` the wrong way, which is the guardrail this keying
    exists to hold.
    """

    if not reference_link_enabled() or not inventory or not inventory.get("available"):
        return ""
    if not reference_link_slot(task):
        return ""
    urls = list(inventory.get("urls") or ())
    if not urls:
        return ""
    key = "|".join(
        str(getattr(task, name, "") or "")
        for name in ("real_sample_id", "local_task_id", "branch_id", "claim_key")
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return urls[int(digest, 16) % len(urls)]


def reference_link_offer(url: str) -> str:
    """Render the Writer cue.

    E4: naming the concrete token gets about 1.0 compliance where naming the
    category gets 0.23. The category cue is what shipped -- "make it a short
    reference or link-like aside" -- and it produced zero links across 1,974
    slots. This hands over the exact string instead.
    """

    link = str(url or "").strip()
    if not link:
        return ""
    return (
        "This slot's real counterpart carried a link. Include this exact URL once, "
        f"inline, the way a commenter drops a source mid-sentence: {link} "
        "Write it as a bare URL. Do not wrap it in markdown link syntax, do not "
        "describe it, do not add a title for it, and do not write any other URL."
    )
