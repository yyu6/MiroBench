#!/usr/bin/env python3
"""Self-loop reviser: iterate metric-targeted rewrites under a no-regression gate.

    python3 selfloop/controller.py --tags v157_20260903_p0 ... --rounds 6

One round:
  1. score the cohort (official scorers, models held open) and take the verdict
  2. pick the target metric -- the worst one still failing, else the largest |d|
  3. per thread, rank comments by their contribution to that metric and rewrite
     the top few with gpt-5.4-mini, several candidates each
  4. keep a candidate only if it improves that THREAD's own target metric
  5. rescore only the threads that changed, and only the scorers that can move
  6. accept the round only if the target improved and nothing else regressed;
     otherwise roll every thread back to the text it had at the start

The gate is the user's rule, verbatim: "修改就必须只提高 self_bertscore，而不能
让其他任何 metric 下降". Step 6 is where that is enforced, and step 4 only
proposes -- a thread-local gain that costs the cohort is rejected in step 6.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "selfloop"))
import candidate_scorer as C  # noqa: E402
import judge as J  # noqa: E402
import metric_engine as E  # noqa: E402
import reviser as R  # noqa: E402
import selection as SEL  # noqa: E402
import strategies as S  # noqa: E402
import threads as TH  # noqa: E402

RUNS = REPO / "artifacts/generalized_card/runs"


def _report_memory(stage: str) -> float:
    """Resident size, printed at each stage.

    The first full-cohort run was killed with no traceback while holding two
    copies of deberta-xlarge-mnli (7.1 GB). Printing this makes the next such
    death diagnosable from the log instead of by re-deriving it.
    """
    import resource

    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
    print(f"[memory] {stage}: {mb:.0f} MB", flush=True)
    return mb

# Which scorers a metric's value can possibly depend on. Rescoring only these
# is exact: the reply tree never changes, so the structural scorer's output is
# byte-identical, and a metric's own JSON is the only input to its summary row.
SCORER_FOR = {
    "semantic_mean_cosine": ("semantic_uniformity_results.json",),
    "self_bertscore_mean_f1": ("self_bertscore_results.json",),
    "self_bleu_4": ("self_bleu_results.json",),
    "polite_rate": ("politeness_results.json",),
    "impolite_rate": ("politeness_results.json",),
    "neutral_rate": ("politeness_results.json",),
    "mean_story_probability": ("storyseeker_results.json",),
    "emotion_entropy": ("go_emotions_results.json",),
    "hard_disagree_rate": ("stance_disagreement_results.json",),
    "length_cv": ("thread_structure_results.json",),
    "avg_depth": ("thread_structure_results.json",),
    "structural_virality": ("thread_structure_results.json",),
}
# Every scorer whose output can move when comment TEXT changes. `thread_structure`
# is excluded on purpose and `test_structural_metrics_are_invariant` proves it.
TEXT_SENSITIVE = tuple(
    sorted({name for metric, names in SCORER_FOR.items() for name in names
            if metric not in J.STRUCTURAL})
)


@dataclass
class ThreadState:
    tag: str
    work: Path
    thread: Any
    row: dict[str, Any]
    real: dict[str, Any]


def stage(tags: list[str], out: Path, *, force: bool) -> list[ThreadState]:
    """Copy each run's cleaned artifact into the loop's own workspace."""
    import csv

    states = []
    for tag in tags:
        src = RUNS / tag / "cleaned/run_00_sampled_reddit"
        if not src.exists():
            print(f"[skip] {tag}: no cleaned artifact", flush=True)
            continue
        real_csv = RUNS / tag / "matched_evaluation/matched_real_thread_scores.csv"
        if not real_csv.exists():
            print(f"[skip] {tag}: not matched-evaluated yet", flush=True)
            continue
        real_row = next((r for r in csv.DictReader(real_csv.open())
                         if not r["thread_id"].startswith("__")), None)
        if real_row is None:
            print(f"[skip] {tag}: matched real row missing", flush=True)
            continue
        work = out / tag
        if work.exists() and force:
            shutil.rmtree(work)
        if not work.exists():
            work.mkdir(parents=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, work / f.name)
        states.append(ThreadState(tag=tag, work=work, thread=TH.load(work),
                                  row={}, real=real_row))
    return states


def rescore(state: ThreadState, *, only: tuple[str, ...], device: str) -> None:
    state.row = E.score_run_dir(state.work, device=device, only=only, force=True)


def cohort_verdict(states: list[ThreadState]) -> dict[str, J.MetricVerdict]:
    return J.verdict([s.row for s in states], [s.real for s in states])


def thread_target(state: ThreadState, metric: str) -> float:
    """This thread's own matched real value for the metric.

    For `self_bertscore_mean_f1` the local objective is a proxy on a different
    scale, so the target has to be the proxy computed on the real thread's
    counterpart quantities rather than its BERTScore. `real_proxy_target`
    supplies that; here it is only the plain lookup.
    """
    try:
        return float(state.real.get(metric))
    except (TypeError, ValueError):
        return 0.0


def local_target(state: ThreadState, metric: str) -> float:
    if metric != "self_bertscore_mean_f1":
        return thread_target(state, metric)
    # The proxy is 0.5*self_bleu + 0.5*semantic, both of which the matched real
    # row carries, so the real thread's proxy value is directly available.
    lexical = _real_value(state, "self_bleu_4")
    semantic = _real_value(state, "semantic_mean_cosine")
    return 0.5 * lexical + 0.5 * semantic


def _real_value(state: ThreadState, key: str) -> float:
    try:
        return float(state.real.get(key))
    except (TypeError, ValueError):
        return 0.0


def local_score(cache: C.ThreadCache, guard: C.GuardCache, metric: str,
                index: int, candidate: str | None = None) -> float:
    """The thread's target-metric value, with `index` optionally swapped.

    Everything routes through the two caches so a candidate costs a rank-one
    update rather than a rescore. Returning NaN for an unhandled metric is not
    an option: `gain = base - abs(nan - want)` is NaN, `NaN <= 0` is False and
    `cost > best` is False, so the round silently applies nothing. That is
    exactly what round 5 did on `emotion_entropy` -- applied=0 with the API
    already paid for -- before GuardCache was wired in here.
    """
    if metric == "semantic_mean_cosine":
        return (C.semantic_mean_cosine(cache.vectors) if candidate is None
                else cache.semantic_if(index, candidate))
    if metric == "self_bleu_4":
        return (cache.bleu_total / cache.pair_count if cache.pair_count else 0.0
                ) if candidate is None else cache.self_bleu_if(index, candidate)
    if metric == "self_bertscore_mean_f1":
        # Proxy, not the model. One `bert_pair_f1` call scores n-1 pairs of
        # deberta-xlarge; a round evaluates ~135 candidates, and every attempt
        # at this round was SIGKILLed mid-loop. BERTScore is greedily aligned
        # token similarity, and lexical overlap plus embedding similarity rank
        # the same comments: Spearman +0.761 against the real per-comment
        # ordering on a 42-comment thread (2026-09-04).
        #
        # This ranks candidates. The round is still gated on the official
        # self_bertscore scorer, so a proxy that mis-ranks costs search quality
        # and can never let a regression through.
        lexical = (cache.bleu_total / cache.pair_count if cache.pair_count else 0.0
                   ) if candidate is None else cache.self_bleu_if(index, candidate)
        semantic = (C.semantic_mean_cosine(cache.vectors) if candidate is None
                    else cache.semantic_if(index, candidate))
        return 0.5 * lexical + 0.5 * semantic
    values = guard.values(index, candidate) if candidate is not None else guard.values()
    if metric in values:
        return values[metric]
    # hard_disagree_rate is pairwise over the parent/child pairs of the whole
    # thread; there is no per-comment form, so it is never a local objective.
    return float("nan")


# Metrics a round can actually optimise. `hard_disagree_rate` is excluded
# because `local_score` cannot evaluate a candidate for it, and a target with
# no local score applies nothing while still paying for the API calls.
REVISABLE = tuple(m for m in S.STRATEGIES if m != "hard_disagree_rate")


# Cheap enough to evaluate on every candidate, and the two that a rewrite is
# most likely to disturb: pushing text apart lowers lexical overlap, and
# changing what a comment says moves its semantic distance to its neighbours.
# G181 measured self_bleu_4 moving the wrong way whenever comments are pushed
# apart, and the loop's first live round rejected on exactly that.
# Every metric a text rewrite can move and that is cheap enough to evaluate on
# each candidate. `self_bertscore_mean_f1` and `hard_disagree_rate` are absent
# because both are pairwise over the whole thread; they stay protected by the
# round gate. The loop's first full round was rejected for moving story and
# emotion, neither of which was guarded then.
GUARD_METRICS = ("self_bleu_4", "semantic_mean_cosine", "mean_story_probability",
                 "polite_rate", "impolite_rate", "neutral_rate", "emotion_entropy",
                 "length_cv")


def guard_values(cache: C.ThreadCache, guard: C.GuardCache,
                 index: int | None = None, candidate: str | None = None) -> dict[str, float]:
    if candidate is None:
        pairwise = {
            "self_bleu_4": cache.bleu_total / cache.pair_count if cache.pair_count else 0.0,
            "semantic_mean_cosine": C.semantic_mean_cosine(cache.vectors),
        }
    else:
        pairwise = {"self_bleu_4": cache.self_bleu_if(index, candidate),
                    "semantic_mean_cosine": cache.semantic_if(index, candidate)}
    return {**pairwise, **guard.values(index, candidate)}


def guard_penalty(before: dict[str, float], after: dict[str, float],
                  real: dict[str, Any], *, skip: str) -> float:
    """How much a candidate moves the guard metrics AWAY from this thread's real
    counterpart. Zero when it moves them toward it, so a candidate that helps
    two metrics at once is preferred rather than merely tolerated."""
    total = 0.0
    for key in GUARD_METRICS:
        if key == skip:
            continue
        try:
            want = float(real.get(key))
        except (TypeError, ValueError):
            continue
        scale = abs(want) or 1.0
        moved = (abs(after[key] - want) - abs(before[key] - want)) / scale
        total += max(0.0, moved)
    return total


def choose_subset(states: list[ThreadState], before: dict[str, dict],
                  changed: list[ThreadState], before_v: dict[str, J.MetricVerdict],
                  metric: str) -> tuple[set[str], list[str], bool]:
    """Largest subset of the revised threads that gains without regressing.

    Once a thread is scored, trying a different subset costs a statistics
    recompute and no model call at all -- so an all-or-nothing round throws away
    work for free. Greedy: start with every revised thread, and while a
    non-target metric regresses, drop whichever thread's removal repairs it
    most. Terminates because each step removes one thread.
    """
    keep = {state.tag for state in changed}

    def verdict_for(subset: set[str]) -> dict[str, J.MetricVerdict]:
        rows = [state.row if state.tag in subset else before[state.tag]
                for state in states]
        return J.verdict(rows, [state.real for state in states])

    for _ in range(len(changed) + 1):
        current = verdict_for(keep)
        hurt = J.regressions(before_v, current, target=metric)
        gained = J.improved(before_v, current, target=metric, min_gain=0.0)
        if not hurt:
            return (keep, [], gained) if gained else (set(), [], False)
        if not keep:
            return set(), hurt, False
        # Drop the thread whose removal leaves the fewest regressions, breaking
        # ties by the one that keeps the target gain highest.
        best_tag, best_key = None, None
        for tag in sorted(keep):
            trial = keep - {tag}
            trial_v = verdict_for(trial)
            key = (len(J.regressions(before_v, trial_v, target=metric)),
                   -(trial_v[metric].quality() if metric in trial_v else 0.0))
            if best_key is None or key < best_key:
                best_tag, best_key = tag, key
        keep.discard(best_tag)
    return set(), ["subset_search_exhausted"], False


def run_round(states: list[ThreadState], *, api, model: str, metric: str,
              community: str, protected: list[str], device: str,
              round_idx: int, workers: int, verbose: bool) -> dict[str, Any]:
    strategy = S.STRATEGIES[metric]
    before = {s.tag: dict(s.row) for s in states}
    saved = {s.tag: TH.snapshot(s.thread) for s in states}

    targets: list[R.Target] = []
    measured, targets_by_thread, local_targets = {}, {}, {}
    for state in states:
        texts = state.thread.scored_texts
        value = float(state.row.get(metric) or 0.0)
        want = thread_target(state, metric)
        measured[state.tag] = value
        targets_by_thread[state.tag] = want
        local_targets[state.tag] = local_target(state, metric)
        if len(texts) < 4:
            continue
        order = SEL.rank(texts, metric, too_high=value > want)
        k = SEL.budget(len(texts), strategy.max_share)
        for position in order[:k]:
            node_index = state.thread.scored[position]
            targets.append(R.Target(
                thread_id=state.tag, index=position,
                comment_id=str(state.thread.nodes[node_index].get("comment_id") or position),
                text=texts[position],
                parent_text=state.thread.parent_text(node_index),
                neighbours=[t for i, t in enumerate(texts) if i != position],
            ))
    if not targets:
        return {"round": round_idx, "metric": metric, "accepted": False,
                "reason": "no_targets", "targets": 0, "applied": 0,
                "threads_changed": 0, "api_seconds": 0.0, "score_seconds": 0.0}

    t0 = time.time()
    proposals = R.propose(api, targets, metric=metric, measured=measured,
                          thread_target=targets_by_thread, community=community,
                          protected=protected, model=model,
                          candidates=strategy.candidates, workers=workers)
    api_seconds = time.time() - t0

    applied = 0
    by_tag = {s.tag: s for s in states}
    caches = {s.tag: C.ThreadCache(s.thread.scored_texts)
              for s in states if len(s.thread.scored_texts) >= 4}
    guards = {tag: C.GuardCache(by_tag[tag].thread.scored_texts) for tag in caches}
    proposals = {k: v for k, v in proposals.items() if k[0] in caches}
    for (tag, index), candidates in proposals.items():
        if not candidates:
            continue
        state = by_tag[tag]
        cache = caches[tag]
        guard = guards[tag]
        want = local_targets[tag]
        base_gap = abs(local_score(cache, guard, metric, index) - want)
        base_guard = guard_values(cache, guard)
        best, best_cost = None, 0.0
        for candidate in candidates:
            body = candidate["text"]
            gain = base_gap - abs(local_score(cache, guard, metric, index, body) - want)
            if gain <= 0:
                continue
            # Scale the gain the same way the penalty is scaled, so the two are
            # comparable, then require the candidate to be net positive. A
            # candidate that buys the target by wrecking a guard is dropped
            # here rather than costing the whole round at the gate.
            scaled_gain = gain / (abs(want) or 1.0)
            cost = scaled_gain - guard_penalty(
                base_guard, guard_values(cache, guard, index, body),
                state.real, skip=metric)
            if cost > best_cost:
                best, best_cost = body, cost
        if best is not None:
            state.thread.set_text(state.thread.scored[index], best)
            cache.commit(index, best)
            guard.commit(index, best)
            applied += 1
    if applied == 0:
        return {"round": round_idx, "metric": metric, "accepted": False,
                "reason": "no_candidate_improved_its_thread",
                "targets": len(targets), "applied": 0, "threads_changed": 0,
                "api_seconds": round(api_seconds, 1), "score_seconds": 0.0}

    _report_memory(f"round{round_idx} after candidates")
    changed = []
    for state in states:
        if TH.snapshot(state.thread) != saved[state.tag]:
            TH.save(state.thread)
            changed.append(state)
    t1 = time.time()
    for state in changed:
        rescore(state, only=TEXT_SENSITIVE, device=device)
    score_seconds = time.time() - t1

    before_v = J.verdict([before[s.tag] for s in states], [s.real for s in states])
    keep, hurt, gained = choose_subset(states, before, changed, before_v, metric)
    accepted = bool(keep) and gained

    # Roll back every thread that is not in the kept subset. Threads outside
    # `changed` were never edited, so this is a no-op for them.
    dropped = []
    for state in changed:
        if state.tag not in keep:
            TH.restore(state.thread, saved[state.tag])
            state.row = before[state.tag]
            dropped.append(state.tag)
    if not accepted:
        for state in states:
            TH.restore(state.thread, saved[state.tag])
            state.row = before[state.tag]
    after_v = cohort_verdict(states)
    return {
        "round": round_idx, "metric": metric, "accepted": accepted,
        "reason": "" if accepted else ("no_gain" if not hurt else "regressed:" + ",".join(hurt)),
        "targets": len(targets), "applied": applied,
        "threads_changed": len(changed),
        "threads_kept": sorted(keep), "threads_dropped": sorted(dropped),
        "api_seconds": round(api_seconds, 1), "score_seconds": round(score_seconds, 1),
        "before": {k: round(v.d, 4) for k, v in before_v.items()},
        "after": {k: round(v.d, 4) for k, v in after_v.items()},
        "pass_before": J.pass_count(before_v), "pass_after": J.pass_count(after_v),
    }


def _next_target(verdicts: dict[str, J.MetricVerdict], tried: set[str]) -> str:
    """Worst failing metric, skipping ones that already failed to move twice."""
    ranked = sorted(verdicts.items(), key=lambda kv: (kv[1].passes, -abs(kv[1].d)))
    for key, _ in ranked:
        if key in REVISABLE and key not in tried:
            return key
    return ""


def _checkpoint(states: list[ThreadState], out: Path, round_idx: int,
                tried: set[str], attempts: dict[str, int]) -> None:
    payload = {"round": round_idx, "tried": sorted(tried),
               "attempts": attempts,
               "threads": {s.tag: {"texts": s.thread.scored_texts, "row": s.row}
                           for s in states}}
    tmp = out / "checkpoint.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(out / "checkpoint.json")


def _resume(states: list[ThreadState], out: Path) -> tuple[int, set[str], dict[str, int]]:
    """Restore the text and scores a previous process reached.

    The loop is long-running and holds ~8 GB, and this machine has killed it
    between rounds with no traceback and no crash report more than once. A run
    that has to restart from round 1 wastes every API call it already paid for,
    so the checkpoint is written after every round and read here.
    """
    path = out / "checkpoint.json"
    if not path.exists():
        return 1, set(), {}
    payload = json.loads(path.read_text())
    by_tag = {s.tag: s for s in states}
    for tag, saved in (payload.get("threads") or {}).items():
        state = by_tag.get(tag)
        if state is None:
            continue
        texts = saved.get("texts") or []
        if len(texts) == len(state.thread.scored):
            TH.restore(state.thread, texts)
        if saved.get("row"):
            state.row = saved["row"]
    resumed = int(payload.get("round") or 0) + 1
    tried = set(payload.get("tried") or [])
    attempts = {str(k): int(v) for k, v in (payload.get("attempts") or {}).items()}
    print(f"[resume] checkpoint found; continuing at round {resumed}"
          f"{'  already tried: ' + ','.join(sorted(tried)) if tried else ''}", flush=True)
    return resumed, tried, attempts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/selfloop")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--metric", default="", help="fix the target instead of auto-picking")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domain", default="celebrity_geo")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="an earlier --out directory to continue from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    domain_config = json.loads(
        (REPO / f"generalized_card/configs/domains/{args.domain}.json").read_text())
    community = str(domain_config.get("community_context") or "Reddit")
    protected = list(domain_config.get("protected_entity_terms") or [])

    out = args.resume_from or (args.out / time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    states = stage(args.tags, out, force=args.force)
    if len(states) < 3:
        raise SystemExit("need at least 3 matched-evaluated threads")

    print(f"[selfloop] {len(states)} threads  model={args.model}  out={out}", flush=True)
    first_round, resumed_tried, attempts = _resume(states, out)
    t0 = time.time()
    for state in states:
        if not state.row:
            rescore(state, only=(), device=args.device)
    _report_memory("baseline")
    baseline = cohort_verdict(states)
    print(f"[selfloop] baseline scored in {time.time()-t0:.0f}s\n")
    print(J.render(baseline, len(states)), flush=True)
    (out / "baseline.json").write_text(json.dumps(
        {k: asdict(v) for k, v in baseline.items()}, indent=1))

    if args.dry_run:
        print(f"\n[dry-run] would target: {J.worst_failing(baseline)}")
        return

    api = R.client()
    history = []
    tried: set[str] = set(resumed_tried)
    for round_idx in range(first_round, args.rounds + 1):
        current = cohort_verdict(states)
        metric = args.metric or _next_target(current, tried)
        if not metric or metric not in S.STRATEGIES:
            print(f"[stop] no metric left to try"); break
        # A metric whose round has been killed twice will be killed a third
        # time; retire it rather than spend the run restarting into it.
        attempts[metric] = attempts.get(metric, 0) + 1
        if attempts[metric] > 2:
            print(f"[skip] {metric}: {attempts[metric] - 1} attempts already died", flush=True)
            tried.add(metric)
            _checkpoint(states, out, round_idx - 1, tried, attempts)
            continue
        _checkpoint(states, out, round_idx - 1, tried, attempts)
        print(f"\n===== round {round_idx}  target={metric}  "
              f"d={current[metric].d:+.2f} {'PASS' if current[metric].passes else 'FAIL'} =====",
              flush=True)
        try:
            result = run_round(states, api=api, model=args.model, metric=metric,
                               community=community, protected=protected,
                               device=args.device, round_idx=round_idx,
                               workers=args.workers, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001 - a bad round must not end the run
            import traceback

            traceback.print_exc()
            result = {"round": round_idx, "metric": metric, "accepted": False,
                      "reason": f"exception:{type(exc).__name__}: {exc}"}
        # A metric that yielded nothing is not worth an immediate retry: the
        # same selection and the same instruction would re-roll the same dice.
        # Move the budget to the next worst metric, and reopen everything as
        # soon as a round is accepted, since the cohort has changed.
        if result.get("accepted"):
            tried.clear()
            attempts.clear()
        else:
            tried.add(metric)
        # The round completed, so it did not die; clear its death count.
        attempts[metric] = 0
        # Persist the revised text after every round, so a crash costs one
        # round rather than the whole run.
        _checkpoint(states, out, round_idx, tried, attempts)
        history.append(result)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[round {round_idx}] {'ACCEPTED' if result['accepted'] else 'REJECTED'} "
              f"{result.get('reason','')}  applied={result.get('applied',0)}"
              f"  api={result.get('api_seconds',0)}s score={result.get('score_seconds',0)}s"
              f"  pass {result.get('pass_before','?')}->{result.get('pass_after','?')}",
              flush=True)

    final = cohort_verdict(states)
    print("\n===== FINAL =====")
    print(J.render(final, len(states)))
    (out / "final.json").write_text(json.dumps(
        {k: asdict(v) for k, v in final.items()}, indent=1))
    print(f"\nartifacts: {out}")


if __name__ == "__main__":
    main()
