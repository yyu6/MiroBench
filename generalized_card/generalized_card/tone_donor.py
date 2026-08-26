"""Drawn appreciative sentences for polite-assigned slots (v120).

`polite_rate` is decided by whether a comment contains **one whole sentence** that
Polite Guard reads as unambiguously appreciative (G6, G53). Six realization-side
mechanisms are dead: more register cues, the omitted conjunction, hedging, length
repair, the bare-assertion frame, and the polite lexicon -- the last decisively,
because generated already carries real's top-45 polite-discriminative tokens at
**1.14x** real prevalence and still converts at 0.26-0.45x (G58). The generator
does not lack the words. It does not produce the sentence.

G53 left exactly one mechanism alive, measured and unbuilt: inserting a single
real short appreciative sentence into a generated non-polite comment flips its
label **0.29-0.50** of the time, against 0.121 for a non-polite real donor. This
module draws that sentence.

Why this is not the lexicon route wearing a different hat: a cue naming a *word*
or a *speech act* leaves the Writer to compose the sentence, and composing it is
the step that fails. This hands over the finished sentence, which is E4's
distinction -- naming the concrete token buys ~1.0 compliance where naming the
category buys 0.23, the same reason `reference_link` hands over a URL string
rather than asking for "a link".

**What it is worth.** With the flip rate at the LOW end of G53's range the polite
row of the realization matrix moves 0.384 -> 0.563, and `polite_rate` at N=150
goes from P(pass) 0.17 to **0.90** raw (0.64 -> 0.99 Holm), with `impolite_rate`
0.92 and `neutral_rate` 0.95. All three tone metrics pass. Nothing else measured
comes close, and the tone trio is the only group at literally zero today (G69).

**The risk, named because it is real (G37).** Shared prescribed text converges the
pairwise metrics: slots given the same speech act in v109 scored +0.0255 on
`self_bertscore`. The mitigations are the ones the link arm already validates --
a large inventory, a per-slot hash draw so collisions are rare, and no instruction
beyond the sentence itself. `analysis/tone_carrier/donor_collision_risk.py` prices
the residual on the real scorer.

The inventory is built from **evaluation-excluded threads only** by
`analysis/tone_carrier/harvest_donor_sentences.py`, filtered to be topic-free -- no
product designator, no digit, no URL -- so no evaluation text and no other thread's
subject can reach the output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# **DO NOT SWITCH THIS ON without reading G79/G80/G82.** Measured at N=10 on the
# evaluation pool, the arm works mechanically -- P(realize polite | assign polite)
# 0.397 -> 0.784, exactly what G53 predicted -- and is NET HARMFUL: seven of nine
# metrics got worse, `self_bertscore` FAILED (+2.4% -> +5.6%, Cliff 0.98), and
# 44.4% of comments opened with a prescribed appreciative sentence against real's
# 8.9%. Stripping the donor text back out does NOT recover the gap, so the damage
# is the register the cue induces in the whole comment, not the shared sentence.
# Kept in the tree because the mechanism is the only one that moves `polite_rate`,
# and a redesign that does not dictate the opening has not been tried.
TONE_DONOR_MODE = "off"

# The tone the arm routes on. Only slots the Planner assigned `polite` are
# offered a donor; every other assignment is untouched, so the arm cannot move
# the impolite or neutral rows of the matrix it was priced against.
ROUTED_TONES = ("polite",)

PROFILE_DIR = Path(__file__).resolve().parents[1] / "profiles"


def set_tone_donor_mode(mode: str) -> str:
    """Select the drawn-donor-sentence arm and return its value."""

    global TONE_DONOR_MODE
    value = str(mode or "off").strip().lower()
    TONE_DONOR_MODE = "measured" if value == "measured" else "off"
    return TONE_DONOR_MODE


def tone_donor_enabled() -> bool:
    return TONE_DONOR_MODE == "measured"


def domain_of(profile: dict[str, Any] | None) -> str:
    """The domain key a profile is built under.

    The real profile writes **`domain_id`**. Reading `domain` instead returned ""
    for every slot, `load_donor_inventory("")` found no file, and the arm rendered
    nothing across a whole paid run while `run_config.json` recorded
    `tone_donor: measured` -- E9's silently-inert gate, and it survived a
    verification pass because that pass built its own `{"domain": ...}` dict
    instead of reading a profile off disk.
    """

    data = profile or {}
    for key in ("domain_id", "domain"):
        value = str(data.get(key) or "").strip()
        if value:
            return value
    return ""


def load_donor_inventory(domain: str) -> dict[str, Any]:
    """Read the frozen per-domain inventory, or an empty one when absent.

    Absent is not an error *here*: the arm is off by default and a domain with no
    harvested inventory simply never draws. It IS an error when the arm is on --
    see `require_donor_inventory`, which the preflight calls so a paid run cannot
    start with an inert arm.
    """

    path = PROFILE_DIR / f"{str(domain or '').strip()}_donor_sentences.json"
    if not path.is_file():
        return {"available": False, "sentences": [], "sentence_count": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "sentences": [], "sentence_count": 0}
    sentences = [str(s).strip() for s in (payload.get("sentences") or ()) if str(s).strip()]
    payload["sentences"] = sentences
    payload["available"] = bool(sentences)
    payload["sentence_count"] = len(sentences)
    return payload


def require_donor_inventory(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Load the inventory for this profile, or raise when the arm is on and empty.

    E9: a required gate can be silently inert, and the only way to know is to make
    it fail. An arm that is switched on and draws nothing spends a paid run to
    produce the previous version's output.
    """

    domain = domain_of(profile)
    inventory = load_donor_inventory(domain)
    if tone_donor_enabled() and not inventory.get("available"):
        raise RuntimeError(
            "--tone-donor measured is on but no donor inventory was found for "
            f"domain {domain!r} (looked in {PROFILE_DIR}). Build it with "
            "analysis/tone_carrier/harvest_donor_sentences.py, or turn the arm off."
        )
    return inventory


def tone_donor_slot(task: Any) -> bool:
    """Whether this slot was ASSIGNED polite by the Planner.

    Reads the assignment, never the matched comment's text or label.
    """

    value = str(getattr(task, "tone_target", "") or "").strip().lower().replace(" ", "_")
    return value in ROUTED_TONES


def draw_donor_sentence(task: Any, inventory: dict[str, Any] | None) -> str:
    """Draw one appreciative sentence for this slot, deterministically.

    Keyed on the slot's own identity, exactly as `reference_link.draw_reference_link`
    is, so a rerun of the same slot draws the same sentence and two slots in one
    thread rarely collide. Collisions are the G37 risk, so the key spans every
    field that distinguishes a slot.
    """

    if not tone_donor_enabled() or not inventory or not inventory.get("available"):
        return ""
    if not tone_donor_slot(task):
        return ""
    sentences = list(inventory.get("sentences") or ())
    if not sentences:
        return ""
    key = "|".join(
        str(getattr(task, name, "") or "")
        for name in ("real_sample_id", "local_task_id", "branch_id", "claim_key")
    )
    digest = hashlib.sha256(f"donor:{key}".encode("utf-8")).hexdigest()
    return sentences[int(digest, 16) % len(sentences)]


def donor_sentence_offer(sentence: str) -> str:
    """Render the Writer cue.

    Deliberately carries **no instruction beyond the sentence and where to put
    it**. G37 measured that prescribing a speech act converges the slots that
    share it (+0.0255 on `self_bertscore`); the sentence itself is per-slot
    distinct, an added rationale would not be.
    """

    text = str(sentence or "").strip()
    if not text:
        return ""
    return (
        "Open this comment with exactly this sentence, word for word, then write "
        f"the rest of your comment after it: {text}"
    )
