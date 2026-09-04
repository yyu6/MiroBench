#!/usr/bin/env python3
"""The 12-metric verdict, identical to combined_eval.py.

Imported by the controller so a round's accept/reject decision is made with the
exact statistics the project reports. `cliff` and the PASS rule are copied from
`generalized_card/analysis/self_similarity/combined_eval.py`; `test_judge`
asserts this module reproduces that script's table on a scored cohort.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from scipy.stats import ks_2samp, mannwhitneyu

M12 = (
    "self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine",
    "hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate",
    "length_cv", "avg_depth", "structural_virality",
    "mean_story_probability", "emotion_entropy",
)

# Text-only revision cannot move these: they are functions of the reply tree,
# which the reviser never edits. Skipping their scorers is exact, not an
# approximation, and `test_structural_metrics_are_invariant` proves it.
STRUCTURAL = ("avg_depth", "structural_virality")


def cliff(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a) * len(b)
    if not n:
        return float("nan")
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / n


@dataclass(frozen=True)
class MetricVerdict:
    metric: str
    gen: float
    real: float
    mwu: float
    ks: float
    d: float

    @property
    def passes(self) -> bool:
        return self.mwu > 0.05 and self.ks > 0.05

    @property
    def rel(self) -> float:
        return (self.gen - self.real) / self.real if self.real else float("nan")

    def quality(self) -> float:
        """Ordering key for "did this metric get better".

        Status first, because a PASS is the deliverable; then |d|, the effect
        size the N=150 bar is stated in (G101); then the weaker p-value as a
        tie-break. Deliberately NOT the p-value alone: p saturates near 1.0
        while d keeps moving, so a p-led objective goes blind exactly where the
        remaining work is.
        """
        return (2.0 if self.passes else 0.0) - abs(self.d) - 0.001 * (1.0 - min(self.mwu, self.ks))


def verdict(gen_rows: list[dict[str, Any]], real_rows: list[dict[str, Any]],
            metrics: Sequence[str] = M12) -> dict[str, MetricVerdict]:
    out: dict[str, MetricVerdict] = {}
    for key in metrics:
        g = [float(r[key]) for r in gen_rows if _usable(r.get(key))]
        r = [float(x[key]) for x in real_rows if _usable(x.get(key))]
        if len(g) < 3 or len(r) < 3:
            continue
        out[key] = MetricVerdict(
            metric=key, gen=statistics.mean(g), real=statistics.mean(r),
            mwu=float(mannwhitneyu(g, r, alternative="two-sided").pvalue),
            ks=float(ks_2samp(g, r).pvalue), d=cliff(g, r),
        )
    return out


def _usable(value: Any) -> bool:
    if value in (None, "", "nan"):
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def pass_count(v: dict[str, MetricVerdict]) -> int:
    return sum(1 for item in v.values() if item.passes)


def render(v: dict[str, MetricVerdict], n: int) -> str:
    sd = (2 / n) * ((2 * n + 1) / 12) ** 0.5
    lines = [f"pooled N = {n}   sd of Cliff d at this N = {sd:.3f}", ""]
    lines.append(f"{'metric':30}{'gen':>10}{'real':>10}{'rel%':>9}{'mwu':>9}{'ks':>8}{'d':>7}  verdict")
    lines.append("-" * 92)
    for key in M12:
        if key not in v:
            continue
        item = v[key]
        lines.append(
            f"{key:30}{item.gen:>10.4f}{item.real:>10.4f}{100 * item.rel:>+8.1f}%"
            f"{item.mwu:>9.3f}{item.ks:>8.3f}{item.d:>+7.2f}  {'PASS' if item.passes else 'FAIL'}"
        )
    lines.append(f"\nPASS {pass_count(v)}/12 at N={n}")
    return "\n".join(lines)


# |d| is quantized to 1/N^2 -- 0.0625 at N=4, 0.01 at N=10 -- so a change
# smaller than one step is arithmetically impossible and anything at one step
# is a single pair flipping. The tolerance sits just under one step at N=10 so
# a real move is caught and a rounding wobble is not.
D_TOLERANCE = 0.01


def regressions(before: dict[str, MetricVerdict], after: dict[str, MetricVerdict],
                *, targets: Sequence[str], tolerance: float = D_TOLERANCE) -> list[str]:
    """Metrics the round made worse, by the user's stated rule.

    "不能修好一个却改坏更多" -- so a PASS that becomes a FAIL is a regression,
    and so is a real move of |d| away from zero on any metric the round was not
    spending on. The round's own targets are exempt here and judged by
    `improved` instead.

    Deliberately NOT `quality()`: that carries a p-value tie-break, and a
    p-value moves under float noise while |d| cannot. A first version used
    quality() here and rejected a round for `impolite_rate: d +0.25 -> +0.25`,
    a metric that had not moved at all.
    """
    exempt = set(targets)
    out = []
    for key, old in before.items():
        if key in exempt or key not in after:
            continue
        new = after[key]
        if old.passes and not new.passes:
            out.append(f"{key}:PASS->FAIL")
        elif abs(new.d) > abs(old.d) + tolerance:
            out.append(f"{key}:|d| {abs(old.d):.2f}->{abs(new.d):.2f}")
    return out


def group_drift(before: dict[str, MetricVerdict], after: dict[str, MetricVerdict],
                *, targets: Sequence[str], tolerance: float = D_TOLERANCE) -> list[str]:
    """Members of the round's OWN group that moved away from zero.

    `regressions` exempts the targets, which is right for the accept rule but
    left the subset search blind: when a group member drifted it saw no damage
    to repair, so it returned the whole round instead of dropping the threads
    responsible for the drift. This gives it that handle.
    """
    out = []
    for key in targets:
        if key not in before or key not in after:
            continue
        old, new = before[key], after[key]
        if old.passes and not new.passes:
            out.append(f"{key}:PASS->FAIL")
        elif abs(new.d) > abs(old.d) + tolerance:
            out.append(f"{key}:|d| {abs(old.d):.2f}->{abs(new.d):.2f}")
    return out


def improved(before: dict[str, MetricVerdict], after: dict[str, MetricVerdict],
             *, targets: Sequence[str], min_gain: float = 0.0) -> bool:
    """Did the round move what it was spending on, without hurting the rest of
    its own group?

    A group is one objective, not several. The three similarity metrics are
    three readings of the same pairwise redundancy, so a rewrite that helps one
    usually helps all three; summing their |d| is what lets a single round fix
    all three, and scoring them one at a time throws that away.

    Same |d| basis as `regressions`, so accept and reject cannot disagree: no
    member may drift further from zero, no member may fall out of PASS, and
    either the group's total |d| drops or a member has newly passed.
    """
    keys = [k for k in targets if k in before and k in after]
    if not keys:
        return False
    if any(before[k].passes and not after[k].passes for k in keys):
        return False
    if any(abs(after[k].d) > abs(before[k].d) + D_TOLERANCE for k in keys):
        return False
    if any(not before[k].passes and after[k].passes for k in keys):
        return True
    return (sum(abs(after[k].d) for k in keys)
            < sum(abs(before[k].d) for k in keys) - min_gain)
