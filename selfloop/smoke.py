#!/usr/bin/env python3
"""Self-test: take the worst threads, show what the reviser sees, and check the
numbers move toward real.

    python3 selfloop/smoke.py --tags v157_20260903_p0 ... --threads 3          # dry
    python3 selfloop/smoke.py --tags ... --threads 3 --live                    # one paid round

Dry it costs nothing and answers "is the model being told the right thing about
the right comment". `--live` runs one real round on those threads only and
answers "does what comes back actually move the metric toward the real thread".

Two references are printed for every metric, because they answer different
questions. The MATCHED real thread is the one the gate compares against, so it
is what an accepted round has to close on. The real DISTRIBUTION's p10/median/
p90 says whether a thread is inside the range real threads occupy at all --
which is the shape the CARD-era calibrator reported and is easier to read than
a single number.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "selfloop"))
import candidate_scorer as C  # noqa: E402
import controller as CTL  # noqa: E402
import judge as J  # noqa: E402
import reviser as R  # noqa: E402
import selection as SEL  # noqa: E402
import strategies as S  # noqa: E402


def real_distribution(tags: list[str], metric: str) -> list[float]:
    """Every matched real thread's value for a metric, across the cohort."""
    out = []
    for tag in tags:
        path = (CTL.RUNS / tag / "matched_evaluation" / "matched_real_thread_scores.csv")
        if not path.exists():
            continue
        for row in csv.DictReader(path.open()):
            if row["thread_id"].startswith("__"):
                continue
            try:
                out.append(float(row[metric]))
            except (TypeError, ValueError, KeyError):
                pass
    return sorted(out)


def band(values: list[float], value: float) -> tuple[str, float, float, float]:
    if not values:
        return "?", float("nan"), float("nan"), float("nan")
    n = len(values)
    p10, mid, p90 = (values[int(q * (n - 1))] for q in (0.10, 0.50, 0.90))
    status = "too high" if value > p90 else ("too low" if value < p10 else "in range")
    return status, p10, mid, p90


def gap(state: CTL.ThreadState, metrics: tuple[str, ...]) -> float:
    """Normalized distance from this thread's own matched real counterpart."""
    total = 0.0
    for metric in metrics:
        want = CTL.thread_target(state, metric)
        try:
            have = float(state.row.get(metric))
        except (TypeError, ValueError):
            continue
        total += abs(have - want) / (abs(want) or 1.0)
    return total


def show(state: CTL.ThreadState, metrics: tuple[str, ...],
         dist: dict[str, list[float]], label: str) -> None:
    print(f"\n  {label}")
    print(f"    {'metric':26}{'this thread':>13}{'its real':>10}"
          f"{'real p10':>10}{'p50':>8}{'p90':>8}   where")
    for metric in metrics:
        try:
            have = float(state.row.get(metric))
        except (TypeError, ValueError):
            continue
        want = CTL.thread_target(state, metric)
        status, p10, mid, p90 = band(dist[metric], have)
        print(f"    {metric:26}{have:>13.4f}{want:>10.4f}"
              f"{p10:>10.4f}{mid:>8.4f}{p90:>8.4f}   {status}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--target", default="similarity")
    ap.add_argument("--threads", type=int, default=3, help="how many worst threads to inspect")
    ap.add_argument("--domain", default="celebrity_geo")
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--live", action="store_true", help="actually call the API for one round")
    ap.add_argument("--out", type=Path, default=Path("/tmp/selfloop_smoke_live"))
    args = ap.parse_args()

    metrics = S.metrics_of(args.target)
    domain = json.loads((REPO / f"generalized_card/configs/domains/{args.domain}.json").read_text())
    states = CTL.stage(args.tags, args.out, force=True)
    if not states:
        raise SystemExit("nothing staged")
    dist = {m: real_distribution(args.tags, m) for m in metrics}

    worst = sorted(states, key=lambda s: -gap(s, metrics))[:args.threads]
    print(f"[smoke] {len(states)} threads staged; the {len(worst)} furthest from their "
          f"own real counterpart on '{args.target}':")

    for state in worst:
        texts = state.thread.scored_texts
        print(f"\n{'=' * 78}\n{state.tag}   {len(texts)} scored comments")
        show(state, metrics, dist, "BEFORE")
        cache, guard = C.ThreadCache(texts), C.GuardCache(texts)
        real = {m: CTL.thread_target(state, m) for m in metrics}
        order = SEL.rank(texts, args.target, too_high=True, cache=cache, guard=guard, wants=real)
        print("\n    the comments it would spend a call on, worst first:")
        for position, index in enumerate(order[:3]):
            print(f"      #{position + 1}  {texts[index][:150]}")
        print("\n    and this is what the model is told about the first one:")
        print("    " + SEL.evidence(texts, order[0], args.target, position=0,
                                    cache=cache, guard=guard,
                                    wants=real).replace("\n", "\n    "))

    if not args.live:
        print(f"\n[smoke] dry run. Add --live to spend one round on these "
              f"{len(worst)} threads and re-measure.")
        return

    print(f"\n{'=' * 78}\n[smoke] one live round on {len(worst)} threads, model={args.model}")
    before = {s.tag: dict(s.row) for s in worst}
    result = CTL.run_round(worst, api=R.client(), model=args.model, target=args.target,
                           community=str(domain.get("community_context") or "Reddit"),
                           protected=list(domain.get("protected_entity_terms") or []),
                           device="cpu", round_idx=1, workers=8, feedback={}, verbose=True)
    print(f"\n[smoke] targets={result['targets']}  applied={result.get('applied')}  "
          f"accepted={result['accepted']}  {result.get('reason', '')}")

    print("\n[smoke] did each thread move toward its own real counterpart?")
    print("        (measured before the cohort gate, which may still roll it back)")
    print(f"  {'thread':46}{'metric':26}{'before':>9}{'after':>9}{'real':>9}   verdict")
    tally = {"closer": 0, "further": 0, "flat": 0}
    for tag, rows in (result.get("per_thread") or {}).items():
        for metric, (old, new, want) in rows.items():
            moved = abs(old - want) - abs(new - want)
            mark = "closer" if moved > 1e-6 else ("further" if moved < -1e-6 else "flat")
            tally[mark] += 1
            print(f"  {tag[:45]:46}{metric:26}{old:>9.4f}{new:>9.4f}{want:>9.4f}   {mark}")
    print(f"\n[smoke] {tally['closer']} closer / {tally['further']} further / "
          f"{tally['flat']} unmoved, over {len(result.get('per_thread') or {})} edited threads.")
    print("[smoke] cohort d, before -> after:")
    for metric in metrics:
        b, a = result.get("before", {}).get(metric), result.get("after", {}).get(metric)
        if b is not None:
            print(f"    {metric:26}{b:>+8.3f} -> {a:>+8.3f}")


if __name__ == "__main__":
    main()
