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
# Every scorer whose output can move when comment TEXT changes -- which is all
# of them, `thread_structure` included: `avg_depth` and `structural_virality`
# are functions of the reply tree and cannot move, but `length_cv` is scored in
# the same file and is word counts, so it moves whenever a comment is rewritten.
# What the filter buys is dropping the two structural METRICS from the rescore's
# reasoning, not dropping a scorer; `test_structural_metrics_are_invariant`
# proves those two do not move.
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
    """Copy every scored thread of every run into the loop's own workspace.

    A run directory is not one thread. Cohorts generated with `--shard-size 3`
    hold `run_00_sampled_reddit` .. `run_03_sampled_reddit`, one thread each,
    and the matched CSVs carry one row per run in the same order -- which is why
    48 tags evaluate as 106 threads. Reading only `run_00` revised a quarter of
    the cohort while the gate scored all of it; `_source_sim_dir` in the
    generated CSV names each row's directory, so the mapping is read rather
    than assumed.
    """
    import csv

    states: list[ThreadState] = []
    for tag in tags:
        matched = RUNS / tag / "matched_evaluation"
        gen_csv = matched / "matched_generated_thread_scores.csv"
        real_csv = matched / "matched_real_thread_scores.csv"
        if not (gen_csv.exists() and real_csv.exists()):
            print(f"[skip] {tag}: not matched-evaluated yet", flush=True)
            continue
        gen_rows = [r for r in csv.DictReader(gen_csv.open())
                    if not r["thread_id"].startswith("__")]
        real_rows = [r for r in csv.DictReader(real_csv.open())
                     if not r["thread_id"].startswith("__")]
        if len(gen_rows) != len(real_rows):
            print(f"[skip] {tag}: {len(gen_rows)} generated rows against "
                  f"{len(real_rows)} real rows", flush=True)
            continue
        for gen_row, real_row in zip(gen_rows, real_rows):
            sim_dir = str(gen_row.get("_source_sim_dir") or "").strip()
            name = Path(sim_dir).name if sim_dir else "run_00_sampled_reddit"
            src = RUNS / tag / "cleaned" / name
            if not (src / "discussion.json").exists():
                print(f"[skip] {tag}/{name}: no cleaned artifact", flush=True)
                continue
            key = f"{tag}/{name}"
            work = out / tag / name
            if work.exists() and force:
                shutil.rmtree(work)
            if not work.exists():
                work.mkdir(parents=True)
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, work / f.name)
            # The generated row is the cohort's ALREADY-PUBLISHED score for
            # this thread, carrying all twelve metrics. Rescoring 106 threads
            # to recompute numbers the artifact already holds cost ~20 minutes
            # and was where the run kept being killed. The loop rescores a
            # thread the moment it edits it, so the only requirement is that
            # this row is what the project reports -- and it is the same file
            # `combined_eval.py` reads.
            states.append(ThreadState(tag=key, work=work, thread=TH.load(work),
                                      row=dict(gen_row), real=real_row))
    return states


def rescore(state: ThreadState, *, only: tuple[str, ...], device: str) -> None:
    state.row = E.score_run_dir(state.work, device=device, only=only, force=True)


def rescore_all(states: list[ThreadState], *, only: tuple[str, ...], device: str) -> None:
    """Rescore a batch scorer-major, so one model is resident at a time."""
    rows = E.score_run_dirs([s.work for s in states], device=device,
                            only=only, force=True)
    for state in states:
        state.row = rows.get(state.work, state.row)


def cohort_verdict(states: list[ThreadState]) -> dict[str, J.MetricVerdict]:
    return J.verdict([s.row for s in states], [s.real for s in states])


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def dominant_metric(state: ThreadState, metrics: tuple[str, ...],
                    real: dict[str, float]) -> str:
    """The group member this thread is furthest from its real counterpart on.

    It decides which way the round tells the model to move, and it must not be
    the group's first member. With `metrics[0]` -- self_bertscore -- 28 of 106
    celebrity threads got the opposite instruction: self_bertscore sat a hair
    BELOW its real value, so the round asked for "bring it back onto what the
    thread is about" on a thread whose semantic cosine was 64% ABOVE real. The
    gap is normalized so metrics on different scales are comparable.
    """
    best, best_gap = metrics[0], -1.0
    for metric in metrics:
        have, want = _as_float(state.row.get(metric)), real.get(metric, float("nan"))
        if have != have or want != want:
            continue
        gap = abs(have - want) / (abs(want) or 1.0)
        if gap > best_gap:
            best, best_gap = metric, gap
    return best


def thread_target(state: ThreadState, metric: str) -> float:
    """This thread's own matched real value for the metric, as reported."""
    try:
        return float(state.real.get(metric))
    except (TypeError, ValueError):
        return 0.0


def _lexical(cache: C.ThreadCache, index: int, candidate: str | None) -> float:
    """Thread self-BLEU-4, with `index` optionally swapped for `candidate`."""
    if candidate is None:
        return cache.bleu_total / cache.pair_count if cache.pair_count else 0.0
    return cache.self_bleu_if(index, candidate)


def _semantic(cache: C.ThreadCache, index: int, candidate: str | None) -> float:
    """Thread mean pair cosine, with `index` optionally swapped."""
    if candidate is None:
        return C.semantic_mean_cosine(cache.vectors)
    return cache.semantic_if(index, candidate)


def local_score(cache: C.ThreadCache, guard: C.GuardCache, metric: str,
                index: int, candidate: str | None = None) -> float:
    """The thread's target-metric value, with `index` optionally swapped.

    Everything routes through the two caches so a candidate costs a rank-one
    update rather than a rescore. `self_bertscore_mean_f1` deliberately has no
    entry: a cheap stand-in (0.5*self-BLEU + 0.5*cosine) predicts the direction
    of the official metric's change at Spearman +0.279 (p=0.1) and gets the
    sign right on 21 of 36 single-comment swaps -- 58%, which is noise, so
    optimizing it spends the candidate budget on nothing. Lexical-plus-semantic
    stays where it was actually validated, RANKING comments within a thread
    (Spearman +0.761 against per-comment bert_pair_f1), and the metric itself
    is carried by the round gate on the official scorer.

    Returning NaN for an unhandled metric is not
    an option: `gain = base - abs(nan - want)` is NaN, `NaN <= 0` is False and
    `cost > best` is False, so the round silently applies nothing. That is
    exactly what round 5 did on `emotion_entropy` -- applied=0 with the API
    already paid for -- before GuardCache was wired in here.
    """
    if metric == "semantic_mean_cosine":
        return _semantic(cache, index, candidate)
    if metric == "self_bleu_4":
        return _lexical(cache, index, candidate)
    values = guard.values(index, candidate) if candidate is not None else guard.values()
    if metric in values:
        return values[metric]
    # hard_disagree_rate is pairwise over the parent/child pairs of the whole
    # thread; there is no per-comment form, so it is never a local objective.
    return float("nan")


# Metrics a round can actually optimise. `hard_disagree_rate` is excluded
# because `local_score` cannot evaluate a candidate for it, and a target with
# no local score applies nothing while still paying for the API calls.
# `hard_disagree_rate` is pairwise over parent/child pairs and `self_bertscore`
# is pairwise over the whole thread; neither has a per-comment form, so neither
# can be a target on its own -- a target with no local score applies nothing
# while still paying for the API calls. self_bertscore is still fixed, as a
# member of the similarity group, whose objective is carried by the two metrics
# that do have exact per-comment forms.
REVISABLE = tuple(m for m in S.STRATEGIES
                  if m not in ("hard_disagree_rate", "self_bertscore_mean_f1"))

# Groups first, then whatever is left, worst |d| first. The similarity three
# are what "indistinguishable text" means and they are fixed before anything
# else, because the register metrics can be bought with rewrites that make the
# text MORE uniform -- courtesy and narrative arrive as repeated wording -- and
# that quietly costs the three.
PRIORITY = ("similarity", "register")


def composite_gap(cache: C.ThreadCache, guard: C.GuardCache,
                  metrics: tuple[str, ...], wants: dict[str, float],
                  index: int, candidate: str | None = None) -> float:
    """How far this thread sits from its real counterpart across the round's
    metrics, each normalized by its own scale so semantic cosine (~0.2),
    self-BLEU (~0.05) and the BERTScore proxy (~0.3) are comparable and one
    candidate can be ranked on all three at once."""
    total = 0.0
    for metric in metrics:
        want = wants[metric]
        value = local_score(cache, guard, metric, index, candidate)
        if value != value:  # NaN: no per-comment form for this metric
            continue
        total += abs(value - want) / (abs(want) or 1.0)
    return total


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
    """Every guarded metric for this thread, with `index` optionally swapped."""
    return {"self_bleu_4": _lexical(cache, index, candidate),
            "semantic_mean_cosine": _semantic(cache, index, candidate),
            **guard.values(index, candidate)}


def guard_penalty(before: dict[str, float], after: dict[str, float],
                  real: dict[str, Any], *, skip: tuple[str, ...]) -> float:
    """How much a candidate moves the guard metrics AWAY from this thread's real
    counterpart. Zero when it moves them toward it, so a candidate that helps
    two metrics at once is preferred rather than merely tolerated."""
    total = 0.0
    for key in GUARD_METRICS:
        if key in skip:
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
                  metrics: tuple[str, ...]) -> tuple[set[str], list[str], bool]:
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
        # Damage the search can repair by dropping threads: harm to any metric
        # outside the group, and drift within it. The second half was missing,
        # so a round where most threads moved correctly and a few pushed one
        # group member the wrong way was handed back whole instead of trimmed.
        hurt = (J.regressions(before_v, current, targets=metrics)
                + J.group_drift(before_v, current, targets=metrics))
        gained = J.improved(before_v, current, targets=metrics)
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
            key = (len(J.regressions(before_v, trial_v, targets=metrics))
                   + len(J.group_drift(before_v, trial_v, targets=metrics)),
                   -sum(trial_v[m].quality() for m in metrics if m in trial_v))
            if best_key is None or key < best_key:
                best_tag, best_key = tag, key
        keep.discard(best_tag)
    return set(), ["subset_search_exhausted"], False


def run_round(states: list[ThreadState], *, api, model: str, target: str,
              community: str, protected: list[str], device: str,
              round_idx: int, workers: int, feedback: dict[str, str],
              verbose: bool) -> dict[str, Any]:
    strategy = S.strategy_for(target)
    metrics = S.metrics_of(target)
    before = {s.tag: dict(s.row) for s in states}
    saved = {s.tag: TH.snapshot(s.thread) for s in states}

    # Built before the API calls, not after. The prompt's evidence and the
    # ranking that decides what to send both read these, so building them here
    # costs one embedding pass and one O(n^2) BLEU matrix per thread per round
    # where the previous order paid for two of each.
    caches: dict[str, C.ThreadCache] = {}
    guards: dict[str, C.GuardCache] = {}
    for state in states:
        texts = state.thread.scored_texts
        if len(texts) >= 4:
            caches[state.tag] = C.ThreadCache(texts)
            guards[state.tag] = C.GuardCache(texts)

    proposal_targets: list[R.Target] = []
    wants_by_tag: dict[str, dict[str, float]] = {}
    for state in states:
        cache = caches.get(state.tag)
        if cache is None:
            continue
        texts = state.thread.scored_texts
        # This thread's matched real counterpart, one number per metric of the
        # round. It is what the ranking aims at, what the model is shown, and
        # what a candidate is scored against.
        real = {m: thread_target(state, m) for m in metrics}
        wants_by_tag[state.tag] = real
        dominant = dominant_metric(state, metrics, real)
        too_high = _as_float(state.row.get(dominant)) > real[dominant]
        order = SEL.rank(texts, target, too_high=too_high, cache=cache,
                         guard=guards[state.tag], wants=real)
        instruction = strategy.high if too_high else strategy.low
        for position, index in enumerate(order[:SEL.budget(len(texts), strategy.max_share)]):
            node_index = state.thread.scored[index]
            proposal_targets.append(R.Target(
                thread_id=state.tag, index=index,
                comment_id=str(state.thread.nodes[node_index].get("comment_id") or index),
                text=texts[index],
                parent_text=state.thread.parent_text(node_index),
                instruction=instruction,
                evidence=SEL.evidence(texts, index, target, position=position,
                                      cache=cache, guard=guards[state.tag], wants=real),
                anchors=S.anchors_in(texts[index], protected, context=texts),
            ))
    if not proposal_targets:
        return {"round": round_idx, "metric": target, "accepted": False,
                "reason": "no_targets", "targets": 0, "applied": 0,
                "threads_changed": 0, "api_seconds": 0.0, "score_seconds": 0.0}

    t0 = time.time()
    proposals = R.propose(api, proposal_targets, community=community,
                          keep=strategy.keep, model=model,
                          candidates=strategy.candidates, workers=workers,
                          feedback=feedback)
    api_seconds = time.time() - t0

    applied_text: dict[tuple[str, int], str] = {}
    by_tag = {s.tag: s for s in states}
    for (tag, index), candidates in proposals.items():
        if not candidates or tag not in caches:
            continue
        state, cache, guard = by_tag[tag], caches[tag], guards[tag]
        wants = wants_by_tag[tag]
        base_gap = composite_gap(cache, guard, metrics, wants, index)
        base_guard = guard_values(cache, guard)
        best, best_cost = None, 0.0
        for candidate in candidates:
            body = candidate["text"]
            # `composite_gap` is already scale-normalized, so the gain and the
            # penalty are in the same units and a candidate that buys the
            # target by wrecking a guard is dropped here rather than costing
            # the whole round at the gate.
            gain = base_gap - composite_gap(cache, guard, metrics, wants, index, body)
            if gain <= 0:
                continue
            cost = gain - guard_penalty(base_guard,
                                        guard_values(cache, guard, index, body),
                                        state.real, skip=metrics)
            if cost > best_cost:
                best, best_cost = body, cost
        if best is not None:
            state.thread.set_text(state.thread.scored[index], best)
            cache.commit(index, best)
            guard.commit(index, best)
            applied_text[(tag, index)] = best
    if not applied_text:
        return {"round": round_idx, "metric": target, "accepted": False,
                "reason": "no_candidate_improved_its_thread",
                "targets": len(proposal_targets), "applied": 0, "threads_changed": 0,
                "api_seconds": round(api_seconds, 1), "score_seconds": 0.0}

    _report_memory(f"round{round_idx} after candidates")
    changed = []
    for state in states:
        if TH.snapshot(state.thread) != saved[state.tag]:
            TH.save(state.thread)
            changed.append(state)
    # Hand the memory back before the rescore. The caches hold every thread's
    # embeddings and BLEU matrix, and `candidate_scorer` pins the three guard
    # models; the rescore then loads its own, and the sum is what the OS kills.
    caches.clear()
    guards.clear()
    C.release_models()
    E.release()
    t1 = time.time()
    rescore_all(changed, only=TEXT_SENSITIVE, device=device)
    score_seconds = time.time() - t1
    _report_memory(f"round{round_idx} after rescore")

    # What each edited thread's own numbers did, captured BEFORE the gate can
    # roll them back. Without this a rejected round reports nothing at all, and
    # "the rewrites did not move the metric" is indistinguishable from "they
    # moved it and the cohort gate still said no".
    per_thread = {
        s.tag: {m: [_as_float(before[s.tag].get(m)), _as_float(s.row.get(m)),
                    thread_target(s, m)] for m in metrics}
        for s in changed}

    before_v = J.verdict([before[s.tag] for s in states], [s.real for s in states])
    keep, hurt, gained = choose_subset(states, before, changed, before_v, metrics)
    accepted = bool(keep) and gained

    # Roll back every thread outside the kept subset -- which, when the round is
    # rejected, is all of them: `choose_subset` returns an empty set on every
    # failure path, so the next round starts from the last ACCEPTED text, not
    # from this one's. `TH.restore` writes the file, so disk agrees.
    dropped = []
    for state in changed:
        if state.tag not in keep:
            TH.restore(state.thread, saved[state.tag])
            state.row = before[state.tag]
            dropped.append(state.tag)
    # Tell the next attempt on a rolled-back comment what was already tried, so
    # it re-rolls different dice instead of the same ones. A kept comment has
    # no stale advice to carry.
    for (tag, index), body in applied_text.items():
        key = f"{tag}:{index}"
        if tag in keep:
            feedback.pop(key, None)
        else:
            feedback[key] = ("A previous rewrite of this comment was tried and did "
                             "not survive the check. Produce nothing close to it:\n"
                             + body[:300])
    after_v = cohort_verdict(states)
    return {
        "round": round_idx, "metric": target, "accepted": accepted,
        "reason": "" if accepted else ("no_gain" if not hurt else "regressed:" + ",".join(hurt)),
        "targets": len(proposal_targets), "applied": len(applied_text),
        "threads_changed": len(changed),
        "threads_kept": sorted(keep), "threads_dropped": sorted(dropped),
        "api_seconds": round(api_seconds, 1), "score_seconds": round(score_seconds, 1),
        "per_thread": per_thread,
        "before": {k: round(v.d, 4) for k, v in before_v.items()},
        "after": {k: round(v.d, 4) for k, v in after_v.items()},
        "pass_before": J.pass_count(before_v), "pass_after": J.pass_count(after_v),
    }


def _next_target(verdicts: dict[str, J.MetricVerdict], tried: set[str],
                 priority: tuple[str, ...] = PRIORITY) -> str:
    """The group or single metric the next round should spend on.

    A group is taken while any of its members is still failing, in `priority`
    order, and its members are never targeted individually -- a round that
    fixes self_bertscore alone while semantic_mean_cosine is free to drift is
    exactly the sequential behaviour groups exist to replace.
    """
    for name in priority:
        if name in tried:
            continue
        if any(k in verdicts and not verdicts[k].passes for k in S.metrics_of(name)):
            return name
    grouped = {m for name in priority for m in S.metrics_of(name)}
    ranked = sorted(verdicts.items(), key=lambda kv: (kv[1].passes, -abs(kv[1].d)))
    for key, _ in ranked:
        if key in REVISABLE and key not in tried and key not in grouped:
            return key
    return ""


def _checkpoint(states: list[ThreadState], out: Path, round_idx: int,
                tried: set[str], attempts: dict[str, int],
                feedback: dict[str, str]) -> None:
    payload = {"round": round_idx, "tried": sorted(tried),
               "attempts": attempts, "feedback": feedback,
               "threads": {s.tag: {"texts": s.thread.scored_texts, "row": s.row}
                           for s in states}}
    tmp = out / "checkpoint.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(out / "checkpoint.json")


def _resume(states: list[ThreadState], out: Path
            ) -> tuple[int, set[str], dict[str, int], dict[str, str]]:
    """Restore the text and scores a previous process reached.

    The loop is long-running and holds ~8 GB, and this machine has killed it
    between rounds with no traceback and no crash report more than once. A run
    that has to restart from round 1 wastes every API call it already paid for,
    so the checkpoint is written after every round and read here.
    """
    path = out / "checkpoint.json"
    if not path.exists():
        return 1, set(), {}, {}
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
    feedback = {str(k): str(v) for k, v in (payload.get("feedback") or {}).items()}
    print(f"[resume] checkpoint found; continuing at round {resumed}"
          f"{'  already tried: ' + ','.join(sorted(tried)) if tried else ''}", flush=True)
    return resumed, tried, attempts, feedback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/selfloop")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--target", default="",
                    help="fix the round's target (a group name or a metric) "
                         "instead of auto-picking")
    ap.add_argument("--priority", nargs="*", default=list(PRIORITY),
                    help="groups to exhaust before any single metric")
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
    first_round, resumed_tried, attempts, feedback = _resume(states, out)
    t0 = time.time()
    missing = [s for s in states if not s.row]
    for state in missing:
        rescore(state, only=(), device=args.device)
    _report_memory("baseline")
    baseline = cohort_verdict(states)
    print(f"[selfloop] baseline ready in {time.time()-t0:.0f}s "
          f"({len(states) - len(missing)} rows reused from the cohort's own "
          f"matched evaluation, {len(missing)} rescored)\n")
    print(J.render(baseline, len(states)), flush=True)
    (out / "baseline.json").write_text(json.dumps(
        {k: asdict(v) for k, v in baseline.items()}, indent=1))

    priority = tuple(args.priority)
    if args.dry_run:
        print(f"\n[dry-run] would target: {_next_target(baseline, set(), priority)}")
        return

    api = R.client()
    history = []
    tried: set[str] = set(resumed_tried)
    for round_idx in range(first_round, args.rounds + 1):
        current = cohort_verdict(states)
        target = args.target or _next_target(current, tried, priority)
        if not target:
            print("[stop] no target left to try")
            break
        # A target whose round keeps being killed will keep being killed;
        # retire it rather than spend the run restarting into it. The limit is
        # not 2: the kills seen so far were the OS reclaiming ~8 GB during the
        # rescore, which is a property of this machine and not of the target,
        # and retiring `similarity` for that reason would throw away the only
        # group that matters. `score_run_dirs` is the actual fix; this is the
        # margin for it being imperfect.
        attempts[target] = attempts.get(target, 0) + 1
        if attempts[target] > 4:
            print(f"[skip] {target}: {attempts[target] - 1} attempts already died", flush=True)
            tried.add(target)
            _checkpoint(states, out, round_idx - 1, tried, attempts, feedback)
            continue
        _checkpoint(states, out, round_idx - 1, tried, attempts, feedback)
        standing = "  ".join(
            f"{m.replace('_mean_f1', '').replace('_rate', '')} {current[m].d:+.2f}"
            f"{'' if current[m].passes else '!'}"
            for m in S.metrics_of(target) if m in current)
        print(f"\n===== round {round_idx}  target={target}   {standing} =====", flush=True)
        try:
            result = run_round(states, api=api, model=args.model, target=target,
                               community=community, protected=protected,
                               device=args.device, round_idx=round_idx,
                               workers=args.workers, feedback=feedback,
                               verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001 - a bad round must not end the run
            import traceback

            traceback.print_exc()
            result = {"round": round_idx, "metric": target, "accepted": False,
                      "reason": f"exception:{type(exc).__name__}: {exc}"}
        # A target that yielded nothing is not worth an immediate retry with the
        # same selection and the same instruction. Move the budget on, and
        # reopen everything as soon as a round is accepted, since the cohort has
        # changed underneath.
        if result.get("accepted"):
            tried.clear()
            attempts.clear()
        else:
            tried.add(target)
        # The round completed, so it did not die; clear its death count.
        attempts[target] = 0
        # Persist the revised text after every round, so a crash costs one
        # round rather than the whole run.
        _checkpoint(states, out, round_idx, tried, attempts, feedback)
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
