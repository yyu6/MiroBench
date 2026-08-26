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

# `off` reproduces v113 through v116, which drew exactly one URL for every routed
# slot. Measured over 15,559 evaluation-excluded comments (847 carrying one), a
# real carrying comment holds **1.67** URLs, distributed 1:592 2:146 3:39 4+:70,
# and 34.3 URL tokens. The gate wrote 1.00 and 18.0. Routing is NOT the defect --
# the gate routed 4.51% of slots against the matched threads' own 4.92% carrying
# rate -- so this arm changes the count and nothing else.
REFERENCE_LINK_COUNT_MODE = "off"

# v118. `off` reproduces v117, which drew every URL in a multi-link slot
# independently from the whole inventory -- 249 folded hosts, so the four links it
# wrote into one 46-word comment came from four unrelated places and read as a
# wall (`DECISIONS.md` G61). Real does the opposite: measured on the 150-seed
# evaluation-excluded corpus (424 threads, 11,817 comments, 179 carriers holding
# 2+ non-media URLs), a multi-link comment puts every URL on ONE host 0.771 of the
# time at k=2, 0.640 at k=3 and 0.417 at k=4 -- 0.695 pooled over the arm's
# 2<=k<=4 range. This arm draws that, and changes nothing about routing, count, or
# the single-link case.
#
# The two other defects G61 recorded did not survive measurement and are NOT
# addressed here, deliberately -- see `analysis/self_similarity/url_host_coherence.py`
# and FINDINGS s14 for the retraction.
REFERENCE_LINK_HOST_MODE = "off"

# Real's tail runs past this. Four is where the measured distribution stops being
# dense enough to draw from honestly.
MAX_LINKS_PER_SLOT = 4

# `youtu.be` and `youtube.com` are one place to a reader and two netlocs to
# `urlparse`. Folded before hosts are counted or grouped, on both the measuring
# and the drawing side, so the rate an arm draws is the rate that was measured.
HOST_FOLD = {
    "youtu.be": "youtube.com",
    "m.youtube.com": "youtube.com",
    "np.reddit.com": "reddit.com",
    "old.reddit.com": "reddit.com",
    "new.reddit.com": "reddit.com",
    "redd.it": "reddit.com",
    "a.co": "amazon.com",
    "amzn.to": "amazon.com",
}

# Below this many carriers a per-k rate is noise and the pooled rate is used
# instead. At k=3 and k=4 the camera corpus offers 25 and 24.
MIN_HOST_RATE_SAMPLE = 20

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


def set_reference_link_count(mode: str) -> str:
    """Select the drawn-link-count arm and return its value."""

    global REFERENCE_LINK_COUNT_MODE
    value = str(mode or "off").strip().lower()
    REFERENCE_LINK_COUNT_MODE = "measured" if value == "measured" else "off"
    return REFERENCE_LINK_COUNT_MODE


def set_reference_link_host(mode: str) -> str:
    """Select the host-coherence arm and return its value."""

    global REFERENCE_LINK_HOST_MODE
    value = str(mode or "off").strip().lower()
    REFERENCE_LINK_HOST_MODE = "measured" if value == "measured" else "off"
    return REFERENCE_LINK_HOST_MODE


def reference_link_host_enabled() -> bool:
    return REFERENCE_LINK_HOST_MODE == "measured"


def folded_host(url: str) -> str:
    """The host a reader would say a URL points at.

    Not `urlparse().netloc`: `youtu.be` and `www.youtube.com` are the same place.
    Registrable-domain folding is two labels, which is wrong for `co.uk` and right
    for everything this corpus carries in quantity.
    """

    raw = str(url or "")
    lowered = raw.lower()
    if not lowered.startswith("http"):
        raw = "http://" + raw
    netloc = raw.split("//", 1)[-1].split("/", 1)[0].split("?", 1)[0].lower()
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    netloc = netloc.split(":", 1)[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc in HOST_FOLD:
        return HOST_FOLD[netloc]
    parts = netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) > 2 else netloc


def reference_link_count_enabled() -> bool:
    return REFERENCE_LINK_COUNT_MODE == "measured"


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
    per_carrier: list[int] = []
    # (k, distinct folded hosts) for every carrier inside the arm's own 2..MAX
    # range. Carriers past MAX are left out rather than truncated: truncating an
    # 8-URL comment to its first 4 invents a host structure it never had.
    host_runs: list[tuple[int, int]] = []
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
                per_carrier.append(min(len(found), MAX_LINKS_PER_SLOT))
                if 2 <= len(found) <= MAX_LINKS_PER_SLOT:
                    host_runs.append((len(found), len({folded_host(u) for u in found})))
            mentions.extend(found)
    unique = sorted(set(mentions))
    counts = {
        str(value): round(per_carrier.count(value) / len(per_carrier), 6)
        for value in sorted(set(per_carrier))
    } if per_carrier else {}
    same_host_rate: dict[str, float] = {}
    host_samples: dict[str, int] = {}
    for value in range(2, MAX_LINKS_PER_SLOT + 1):
        rows = [distinct for k, distinct in host_runs if k == value]
        host_samples[str(value)] = len(rows)
        if len(rows) >= MIN_HOST_RATE_SAMPLE:
            same_host_rate[str(value)] = round(
                sum(1 for distinct in rows if distinct == 1) / len(rows), 6
            )
    pooled = (
        round(sum(1 for _, distinct in host_runs if distinct == 1) / len(host_runs), 6)
        if host_runs
        else 0.0
    )
    return {
        "available": bool(unique),
        "urls": unique,
        "url_count": len(unique),
        "mention_count": len(mentions),
        "reference_comment_count": comment_count,
        "carrying_comment_share": (carrying / comment_count) if comment_count else 0.0,
        # Conditional on carrying one, so it composes with the routing decision:
        # routing says whether a slot gets a link, this says how many.
        "urls_per_carrier": counts,
        "mean_urls_per_carrier": (
            round(sum(per_carrier) / len(per_carrier), 4) if per_carrier else 0.0
        ),
        # v118. Conditional on carrying k URLs, the share of carriers whose URLs
        # all sit on one folded host. Only k values with enough carriers to
        # estimate are listed; the rest fall back to `same_host_rate_pooled`.
        "same_host_rate": same_host_rate,
        "same_host_rate_pooled": pooled,
        "same_host_sample_counts": host_samples,
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
    digest = hashlib.sha256(_slot_key(task).encode("utf-8")).hexdigest()
    return urls[int(digest, 16) % len(urls)]


def _slot_key(task: Any) -> str:
    return "|".join(
        str(getattr(task, name, "") or "")
        for name in ("real_sample_id", "local_task_id", "branch_id", "claim_key")
    )


def _unit_draw(key: str, namespace: str) -> float:
    """A deterministic uniform(0,1) for this slot, namespaced per decision.

    Namespacing keeps the decisions independent: drawing a rare large count must
    not also decide whether the links share a host.
    """

    digest = hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _same_host_rate(inventory: dict[str, Any] | None, wanted: int) -> float:
    data = inventory or {}
    rates = data.get("same_host_rate") or {}
    value = rates.get(str(wanted))
    if value is None:
        value = data.get("same_host_rate_pooled")
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _one_host_draw(
    task: Any, inventory: dict[str, Any] | None, wanted: int
) -> list[str]:
    """Draw all `wanted` URLs from one host, or return [] to leave the draw alone.

    The host group is chosen weighted by how many URLs it holds, so the marginal
    host distribution stays what the free draw would have produced; only the
    within-slot structure changes. A group must hold `wanted` distinct URLs to be
    eligible -- 200 of the camera inventory's 249 folded hosts hold exactly one.
    """

    key = _slot_key(task)
    if _unit_draw(key, "host") >= _same_host_rate(inventory, wanted):
        return []
    urls = list((inventory or {}).get("urls") or ())
    if not urls:
        return []
    groups: dict[str, list[str]] = {}
    for url in urls:
        groups.setdefault(folded_host(url), []).append(url)
    eligible = sorted(
        (host, members) for host, members in groups.items() if len(members) >= wanted
    )
    if not eligible:
        return []
    total = sum(len(members) for _, members in eligible)
    draw = _unit_draw(key, "hostgroup") * total
    cumulative = 0.0
    chosen = eligible[-1][1]
    for _, members in eligible:
        cumulative += len(members)
        if draw < cumulative:
            chosen = members
            break
    picked: list[str] = []
    index = 0
    while len(picked) < wanted and index < 256:
        digest = hashlib.sha256(f"hosturl:{index}:{key}".encode("utf-8")).hexdigest()
        candidate = chosen[int(digest, 16) % len(chosen)]
        if candidate not in picked:
            picked.append(candidate)
        index += 1
    return picked if len(picked) == wanted else []


def draw_reference_links(task: Any, inventory: dict[str, Any] | None) -> list[str]:
    """Draw this slot's links: one when the count arm is off, N when it is on.

    N is drawn from the inventory's own `urls_per_carrier`, measured on the same
    evaluation-excluded threads the URLs come from. The count digest is
    namespaced away from the URL digest so that drawing a rare first URL is not
    coupled to drawing a large count.

    Each extra URL is drawn without replacement inside the slot. Repeated links
    would be repeated n-grams and would push `self_bleu_4` the wrong way, which
    is the guardrail `draw_reference_link`'s keying already exists to hold.
    """

    first = draw_reference_link(task, inventory)
    if not first:
        return []
    if not reference_link_count_enabled():
        return [first]
    dist = ((inventory or {}).get("urls_per_carrier") or {})
    ordered = sorted(
        ((int(value), float(weight)) for value, weight in dist.items()),
        key=lambda item: item[0],
    )
    total = sum(weight for _, weight in ordered)
    if not ordered or total <= 0:
        return [first]
    key = _slot_key(task)
    draw = _unit_draw(key, "count")
    wanted, cumulative = ordered[-1][0], 0.0
    for value, weight in ordered:
        cumulative += weight / total
        if draw < cumulative:
            wanted = value
            break
    wanted = max(1, min(int(wanted), MAX_LINKS_PER_SLOT))
    if wanted <= 1:
        return [first]
    if reference_link_host_enabled():
        # v118. Replaces the whole list, `first` included: anchoring on a freely
        # drawn URL and then matching its host would honour the rate only for the
        # 79% of inventory URLs that sit in a group big enough, and undershoot.
        one_host = _one_host_draw(task, inventory, wanted)
        if one_host:
            return one_host
    urls = list((inventory or {}).get("urls") or ())
    picked, index = [first], 0
    while len(picked) < wanted and index < 64:
        extra_digest = hashlib.sha256(f"extra:{index}:{key}".encode("utf-8")).hexdigest()
        candidate = urls[int(extra_digest, 16) % len(urls)]
        if candidate not in picked:
            picked.append(candidate)
        index += 1
    return picked


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


def reference_links_offer(urls: list[str] | None) -> str:
    """Render the cue for a drawn list. One link renders `reference_link_offer`
    verbatim, so the count arm is byte-identical wherever it draws one."""

    links = [str(u).strip() for u in (urls or []) if str(u).strip()]
    if not links:
        return ""
    if len(links) == 1:
        return reference_link_offer(links[0])
    joined = "  ".join(links)
    return (
        "This slot's real counterpart carried links. Include these exact URLs, "
        f"{len(links)} of them, inline and in different places, the way a "
        f"commenter drops sources mid-sentence: {joined} "
        "Write each as a bare URL. Do not wrap them in markdown link syntax, do "
        "not describe them, do not add titles for them, and do not write any "
        "other URL."
    )
