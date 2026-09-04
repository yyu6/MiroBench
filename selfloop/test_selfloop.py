#!/usr/bin/env python3
"""Tests for the self-loop reviser. Run: python3 selfloop/test_selfloop.py

The ones that matter are the equivalence tests: this package is only safe to
gate on because its numbers are the official scorers' numbers.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "selfloop"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

import judge as J
import selection as SEL
import strategies as S
import threads as TH

SCORED = REPO / "artifacts/generalized_card/runs/v157_20260903_p5/cleaned/run_00_sampled_reddit"
TAGS = [f"v157_20260903_p{i}" for i in range(10)]


class JudgeTest(unittest.TestCase):
    def test_reproduces_combined_eval(self) -> None:
        """The verdict must equal the script the project reports from."""
        import csv, subprocess

        def rows(which):
            out = []
            for tag in TAGS:
                f = REPO / f"artifacts/generalized_card/runs/{tag}/matched_evaluation/matched_{which}_thread_scores.csv"
                if not f.exists():
                    self.skipTest("cohort not scored")
                out += [r for r in csv.DictReader(f.open()) if not r["thread_id"].startswith("__")]
            return out

        mine = J.verdict(rows("generated"), rows("real"))
        proc = subprocess.run(
            [sys.executable, str(REPO / "generalized_card/analysis/self_similarity/combined_eval.py"),
             "--dedupe", "--tags", *TAGS],
            capture_output=True, text=True, cwd=REPO, check=True)
        official = {}
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) == 8 and parts[-1] in ("PASS", "FAIL"):
                official[parts[0]] = (float(parts[1]), float(parts[4]), float(parts[5]),
                                      float(parts[6]), parts[7])
        self.assertEqual(set(official), set(mine))
        for key, (gen, mwu, ks, d, verdict) in official.items():
            item = mine[key]
            self.assertAlmostEqual(item.gen, gen, places=4, msg=key)
            self.assertAlmostEqual(item.mwu, mwu, places=3, msg=key)
            self.assertAlmostEqual(item.ks, ks, places=3, msg=key)
            self.assertAlmostEqual(item.d, d, places=2, msg=key)
            self.assertEqual("PASS" if item.passes else "FAIL", verdict, msg=key)

    def test_regression_ignores_pvalue_noise(self) -> None:
        mk = lambda d, p: J.MetricVerdict("m", 0, 0, p, p, d)
        before = {"a": mk(0.25, 0.7), "t": mk(0.75, 0.1)}
        # d unchanged, p worse -> not a regression
        self.assertEqual([], J.regressions(before, {"a": mk(0.25, 0.4), "t": mk(0.6, 0.2)}, targets=["t"]))
        # |d| genuinely larger -> a regression
        self.assertEqual(1, len(J.regressions(before, {"a": mk(0.40, 0.7), "t": mk(0.6, 0.2)}, targets=["t"])))
        # PASS -> FAIL is always a regression, even with |d| smaller
        self.assertEqual(1, len(J.regressions(before, {"a": mk(0.20, 0.01), "t": mk(0.6, 0.2)}, targets=["t"])))

    def test_drift_inside_the_group_is_repairable(self) -> None:
        """`regressions` exempts the targets, which is right for the accept rule
        but left the subset search with nothing to drop threads for when a group
        member drifted -- so the whole round was handed back instead of trimmed."""
        mk = lambda d, p: J.MetricVerdict("m", 0, 0, p, p, d)
        before = {"a": mk(0.50, 0.01), "b": mk(0.30, 0.02)}
        after = {"a": mk(0.30, 0.01), "b": mk(0.45, 0.02)}
        self.assertEqual([], J.regressions(before, after, targets=["a", "b"]))
        self.assertEqual(1, len(J.group_drift(before, after, targets=["a", "b"])))

    def test_a_group_is_one_objective(self) -> None:
        """The whole point of a group: one round, three metrics, and a member
        that drifts the wrong way sinks the round even if the sum improves."""
        mk = lambda d, p: J.MetricVerdict("m", 0, 0, p, p, d)
        group = ["a", "b"]
        before = {"a": mk(0.50, 0.01), "b": mk(0.30, 0.02)}
        self.assertTrue(J.improved(before, {"a": mk(0.30, 0.01), "b": mk(0.28, 0.02)},
                                   targets=group))
        # sum falls 0.80 -> 0.65, but `b` moved away from zero
        self.assertFalse(J.improved(before, {"a": mk(0.20, 0.01), "b": mk(0.45, 0.02)},
                                    targets=group))
        # neither member is a regression to the OTHER metrics' gate
        self.assertEqual([], J.regressions(before, {"a": mk(0.9, 0.01), "b": mk(0.9, 0.02)},
                                           targets=group))


class ThreadViewTest(unittest.TestCase):
    def test_scored_view_matches_the_official_loader(self) -> None:
        """Indices must address the same comments the scorers read."""
        from score_thread_semantic_uniformity import load_generated_comments

        for tag in ("p0", "p1", "p5", "p8"):
            src = REPO / f"artifacts/generalized_card/runs/v157_20260903_{tag}/cleaned/run_00_sampled_reddit"
            if not src.exists():
                continue
            thread = TH.load(src)
            by_thread, _ = load_generated_comments(src)
            official = [c.text for rows in by_thread.values() for c in rows]
            self.assertEqual(official, thread.scored_texts, msg=tag)

    def test_edit_and_rollback_restores_every_field(self) -> None:
        work = Path("/tmp/selfloop_rollback")
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        shutil.copy2(SCORED / "discussion.json", work / "discussion.json")
        original = json.loads((work / "discussion.json").read_text())
        thread = TH.load(work)
        saved = TH.snapshot(thread)
        thread.set_text(thread.scored[0], "completely different text")
        TH.save(thread)
        self.assertNotEqual(original, json.loads((work / "discussion.json").read_text()))
        TH.restore(thread, saved)
        restored = json.loads((work / "discussion.json").read_text())
        # word_count is recomputed on edit, so compare the comment bodies and
        # the tree rather than the raw file.
        def bodies(payload):
            out = []
            def walk(cs):
                for c in cs:
                    out.append((c.get("comment_id"), c.get("content"), c.get("parent_comment_id")))
                    walk(c.get("replies") or [])
            walk(payload["posts"][0]["comments"])
            return out
        self.assertEqual(bodies(original), bodies(restored))


class StrategyTest(unittest.TestCase):
    def test_every_revisable_metric_has_a_strategy(self) -> None:
        missing = [m for m in J.M12 if m not in S.STRATEGIES and m not in J.STRUCTURAL]
        self.assertEqual([], missing)

    def test_every_group_member_is_a_real_metric(self) -> None:
        for name, members in S.GROUPS.items():
            self.assertIn(name, S.GROUP_STRATEGY, msg=name)
            for metric in members:
                self.assertIn(metric, J.M12, msg=f"{name}:{metric}")
        overlap = set(S.SIMILARITY) & set(S.REGISTER)
        self.assertEqual(set(), overlap, "a metric in two groups gets two directions")

    def test_no_strategy_names_a_domain(self) -> None:
        """The CARD revisers said 'card/bank/APR/fee/SUB' and could not move domain."""
        banned = ("card", "bank", "apr", "camera", "lens", "laptop", "headphone",
                  "megapixel", "shutter", "phone", "product", "price", "warranty")
        for name, strategy in {**S.STRATEGIES, **S.GROUP_STRATEGY}.items():
            blob = " ".join([strategy.high, strategy.low, strategy.keep]).lower()
            for word in banned:
                self.assertNotIn(f" {word} ", f" {blob} ", msg=f"{name} names '{word}'")
        self.assertNotIn("card", S.SHARED_INVARIANTS.lower())

    def test_anchors_drop_sentence_openers_but_keep_names(self) -> None:
        """Every anchor is an instruction to preserve that span. Listing
        "Honestly" told the model to keep the shared opener self_bleu_4 charges
        for; measured on 12 celebrity cohorts, 1112 of these were being emitted
        across 1159 comments."""
        text = "Honestly the remaster sold me. Sophie said the same thing."
        self.assertEqual([], S.anchors_in(text))
        with_context = S.anchors_in(text, context=["I saw Sophie post about it"])
        self.assertEqual(["Sophie"], with_context)

    def test_anchors_capture_facts_without_a_vocabulary(self) -> None:
        text = 'Paid $1,200 in 2019 and https://x.com/a called it "a total mess" — ask Olivia Rodrigo'
        found = [a.lower() for a in S.anchors_in(text, ["Olivia Rodrigo"])]
        self.assertIn("https://x.com/a", found)
        self.assertIn("a total mess", found)
        self.assertIn("olivia rodrigo", found)
        self.assertTrue(any("1,200" in a or "1" == a for a in found))

    def test_direction_flips_the_instruction(self) -> None:
        high = S.instruction("semantic_mean_cosine", 0.30, 0.17)
        low = S.instruction("semantic_mean_cosine", 0.10, 0.17)
        self.assertNotEqual(high, low)


class SelectionTest(unittest.TestCase):
    def test_budget_is_bounded(self) -> None:
        self.assertEqual(1, SEL.budget(6, 0.15))
        self.assertEqual(3, SEL.budget(20, 0.15))
        self.assertEqual(12, SEL.budget(400, 0.15))

    def test_rank_prefers_the_redundant_comment(self) -> None:
        texts = ["the shirt is a stunt", "the shirt is just a stunt",
                 "the shirt is only a stunt", "minnesota has not won since 1991",
                 "my dog ate my homework this morning"]
        order = SEL.rank(texts, "similarity", too_high=True)
        self.assertIn(order[0], (0, 1, 2))

    def test_evidence_names_the_comment_it_duplicates(self) -> None:
        """The prompt has to say WHICH comment and WHICH words, or the model is
        being told it is redundant and left to guess with what."""
        texts = ["the shirt is a stunt and nothing more",
                 "the shirt is a stunt and nothing else",
                 "minnesota has not won a game since 1991",
                 "my dog ate my homework this morning"]
        block = SEL.evidence(texts, 0, "similarity", position=0)
        self.assertIn(texts[1], block)          # the nearest one, in full
        self.assertNotIn(texts[3], block)       # not an untargeted dump
        self.assertIn("the shirt is a", block)  # the repeated 4-gram itself

    def test_register_ranking_follows_the_gap(self) -> None:
        """Direction is read off the gap, not assumed: the same thread ranks
        opposite comments first depending on which way it has to move."""
        class Guard:
            n = 4
            polite = ["neutral", "neutral", "polite", "impolite"]
            story = [0.1, 0.1, 0.9, 0.1]
            emotion = ["neutral", "neutral", "joy", "anger"]
            words = [5, 5, 5, 5]

            def values(self):
                return {"polite_rate": 0.25, "impolite_rate": 0.25,
                        "neutral_rate": 0.5, "mean_story_probability": 0.3,
                        "emotion_entropy": 1.0, "length_cv": 0.0}

        texts = ["a", "b", "c", "d"]
        too_much_neutral = SEL.rank(texts, "register", too_high=True, guard=Guard(),
                                    wants={"neutral_rate": 0.1})
        self.assertIn(too_much_neutral[0], (0, 1))
        too_little = SEL.rank(texts, "register", too_high=False, guard=Guard(),
                              wants={"neutral_rate": 0.9})
        self.assertIn(too_little[0], (2, 3))


class ScorerEquivalenceTest(unittest.TestCase):
    def test_candidate_scorer_matches_official_numbers(self) -> None:
        import candidate_scorer as C

        thread = TH.load(SCORED)
        texts = thread.scored_texts
        official_bleu = json.loads((SCORED / "self_bleu_results.json").read_text())["threads"][0]["self_bleu_4"]
        official_sem = json.loads((SCORED / "semantic_uniformity_results.json").read_text())["threads"][0]["mean_cosine_similarity"]
        self.assertAlmostEqual(C.self_bleu_4(texts), official_bleu, places=9)
        self.assertAlmostEqual(C.semantic_mean_cosine(C.embed(texts)), official_sem, places=6)


class EngineTest(unittest.TestCase):
    def test_engine_covers_every_official_metric_command(self) -> None:
        """Drift here means the loop silently stops scoring something."""
        sys.path.insert(0, str(REPO / "generalized_card"))
        from generalized_card.thread_metric_suite import METRIC_COMMANDS
        import metric_engine as E

        official = {row[0] for row in METRIC_COMMANDS}
        mine = {row[0] for row in E.SCORERS}
        self.assertEqual(official, mine)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class StoredStateTest(unittest.TestCase):
    """The per-comment values the loop needs are already on disk -- a thread's
    polite_rate is the fraction of its comments the scorer labelled polite, so
    the labels exist. Reading them must give exactly what recomputing gives, or
    the ranking and the guard drift from the gate."""

    def test_stored_state_equals_a_fresh_computation(self) -> None:
        import candidate_scorer as C

        texts = TH.load(SCORED).scored_texts
        stored = C.stored_state(SCORED, texts)
        self.assertIsNotNone(stored, "the scorers' own output should line up")
        fresh = C.GuardCache(texts)
        reused = C.GuardCache(texts, story=stored["story"], polite=stored["polite"],
                              emotion=stored["emotion"])
        a, b = fresh.values(), reused.values()
        for key in a:
            # 6 places, not 9: a transformer forward pass is not bit-reproducible
            # across batchings. Measured on this thread, 3 of 9 comments come
            # back identical and the rest differ by 1e-8 to 4e-7, while every
            # LABEL matches. The stored numbers are the official scorer's own
            # output, so reusing them agrees with the gate more closely than
            # recomputing does, and 4e-7 sits nine orders below the 0.01 the
            # round gate is decided on.
            self.assertAlmostEqual(a[key], b[key], places=6, msg=key)
        self.assertEqual(fresh.polite, reused.polite)
        self.assertEqual(fresh.emotion, reused.emotion)

        computed = C.ThreadCache(texts)
        loaded = C.ThreadCache(texts, vectors=stored["vectors"])
        self.assertAlmostEqual(C.semantic_mean_cosine(computed.vectors),
                               C.semantic_mean_cosine(loaded.vectors), places=6)

    def test_stored_state_refuses_text_it_does_not_describe(self) -> None:
        """The scorer output is only reusable while it describes the text in the
        directory. Silently trusting a stale file would feed the ranking and the
        guard values for comments that no longer exist."""
        import candidate_scorer as C

        texts = list(TH.load(SCORED).scored_texts)
        self.assertIsNotNone(C.stored_state(SCORED, texts))
        texts[0] = "a completely different comment that was never scored here"
        self.assertIsNone(C.stored_state(SCORED, texts))


class ScorerMajorTest(unittest.TestCase):
    """Scoring a cohort one scorer at a time, freeing each model before the
    next, must give exactly the numbers scoring it one thread at a time gives.
    The reordering exists only to cap peak memory: thread-major holds all eight
    models and the OS killed the 106-thread run at ~8 GB, twice."""

    def test_scorer_major_matches_thread_major(self) -> None:
        import metric_engine as E

        only = ("politeness_results.json", "self_bleu_results.json")
        dirs = []
        for i, tag in enumerate(TAGS[:2]):
            src = REPO / f"artifacts/generalized_card/runs/{tag}/cleaned/run_00_sampled_reddit"
            if not (src / "discussion.json").exists():
                self.skipTest("cohort not present")
            work = Path(f"/tmp/selfloop_order_{i}")
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, work / f.name)
            dirs.append(work)

        one_at_a_time = {d: E.score_run_dir(d, device="cpu", only=only, force=True)
                         for d in dirs}
        batched = E.score_run_dirs(dirs, device="cpu", only=only, force=True)
        for d in dirs:
            a, b = one_at_a_time[d], batched[d]
            for key in ("polite_rate", "impolite_rate", "neutral_rate", "self_bleu_4"):
                self.assertAlmostEqual(float(a[key]), float(b[key]), places=9,
                                       msg=f"{d.name}:{key}")

    def test_release_lets_a_model_go(self) -> None:
        """A cache entry deleted while `candidate_scorer` still points at the
        object frees nothing, which is why both sides have to let go."""
        import candidate_scorer as C
        import metric_engine as E

        C.politeness_labels(["this is a fine comment", "and so is this one"])
        self.assertGreater(E.loaded_models(), 0, "the guard must go through the engine")
        C.release_models()
        E.release()
        self.assertEqual(0, E.loaded_models())
        self.assertIsNone(C._POLITE)


class IncrementalTest(unittest.TestCase):
    """The rank-one updates must equal the full recompute, or the loop is
    ranking candidates on a different quantity from the one it gates on."""

    def test_incremental_matches_full(self) -> None:
        import candidate_scorer as C

        thread = TH.load(SCORED)
        texts = thread.scored_texts
        cache = C.ThreadCache(texts)
        candidate = "Completely different sentence about something else entirely."
        for index in (0, len(texts) // 2, len(texts) - 1):
            trial = list(texts)
            trial[index] = candidate
            self.assertAlmostEqual(
                C.semantic_mean_cosine(C.embed(trial)),
                cache.semantic_if(index, candidate), places=5, msg=f"semantic@{index}")
            self.assertAlmostEqual(
                C.self_bleu_4(trial), cache.self_bleu_if(index, candidate),
                places=9, msg=f"selfbleu@{index}")

    def test_commit_then_incremental_stays_exact(self) -> None:
        """Two edits in a row: the cache must not drift after the first commit."""
        import candidate_scorer as C

        thread = TH.load(SCORED)
        texts = list(thread.scored_texts)
        cache = C.ThreadCache(texts)
        first, second = "A wholly unrelated first rewrite.", "And a second unrelated one."
        cache.commit(0, first)
        texts[0] = first
        texts[1] = second
        self.assertAlmostEqual(C.self_bleu_4(texts), cache.self_bleu_if(1, second), places=9)
        self.assertAlmostEqual(C.semantic_mean_cosine(C.embed(texts)),
                               cache.semantic_if(1, second), places=5)


class StructuralInvarianceTest(unittest.TestCase):
    def test_structural_metrics_are_invariant_to_text(self) -> None:
        """Justifies skipping thread_structure when rescoring a revised thread."""
        import metric_engine as E

        work = Path("/tmp/selfloop_structural")
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for f in SCORED.iterdir():
            if f.is_file():
                shutil.copy2(f, work / f.name)
        before = E.score_run_dir(work, only=("thread_structure_results.json",), force=True)
        thread = TH.load(work)
        for i in thread.scored:
            thread.set_text(i, "totally different words here for every single comment")
        TH.save(thread)
        after = E.score_run_dir(work, only=("thread_structure_results.json",), force=True)
        for key in ("avg_depth", "structural_virality"):
            self.assertEqual(before[key], after[key], msg=key)


class GuardCacheTest(unittest.TestCase):
    """The guard must equal the official per-comment aggregations, or a
    candidate it clears can still fail the round gate."""

    def test_guard_matches_official_thread_rows(self) -> None:
        import candidate_scorer as C

        texts = TH.load(SCORED).scored_texts
        guard = C.GuardCache(texts)
        values = guard.values()
        story = json.loads((SCORED / "storyseeker_results.json").read_text())["threads"][0]
        emotion = json.loads((SCORED / "go_emotions_results.json").read_text())["threads"][0]
        polite = json.loads((SCORED / "politeness_results.json").read_text())["threads"][0]
        self.assertAlmostEqual(values["mean_story_probability"],
                               story["mean_story_probability"], places=6)
        self.assertAlmostEqual(values["emotion_entropy"], emotion["emotion_entropy"], places=9)
        for key in ("polite_rate", "impolite_rate", "neutral_rate"):
            self.assertAlmostEqual(values[key], polite[key], places=9, msg=key)

    def test_guard_swap_equals_a_full_recompute(self) -> None:
        import candidate_scorer as C

        texts = list(TH.load(SCORED).scored_texts)
        guard = C.GuardCache(texts)
        candidate = "Honestly this whole thing just made me laugh out loud."
        swapped = list(texts)
        swapped[1] = candidate
        full = C.GuardCache(swapped).values()
        incremental = guard.values(1, candidate)
        self.assertEqual(set(full), set(incremental))
        for key in full:
            self.assertAlmostEqual(full[key], incremental[key], places=6, msg=key)

    def test_guard_covers_every_cheap_metric(self) -> None:
        """Anything not guarded here is protected only by the round gate."""
        sys.path.insert(0, str(REPO / "selfloop"))
        import controller as CTL

        for metric in CTL.GUARD_METRICS:
            self.assertIn(metric, J.M12)
        unguarded = [m for m in J.M12
                     if m not in CTL.GUARD_METRICS and m not in J.STRUCTURAL]
        # Both remaining ones are pairwise over the whole thread, so a
        # per-candidate guard would cost a rescore each.
        self.assertEqual({"self_bertscore_mean_f1", "hard_disagree_rate"}, set(unguarded))


class LeaveOneOutTest(unittest.TestCase):
    def test_leave_one_out_matches_rescore(self) -> None:
        """The O(n^2) form must equal dropping the comment and rescoring."""
        import candidate_scorer as C
        import selection as SEL

        texts = TH.load(SCORED).scored_texts
        fast = SEL.self_bleu_contributions(texts)
        full = C.self_bleu_4(texts)
        for i in range(len(texts)):
            slow = full - C.self_bleu_4(texts[:i] + texts[i + 1:])
            self.assertAlmostEqual(fast[i], slow, places=9, msg=f"comment {i}")


class LocalScoreTest(unittest.TestCase):
    """A target with no local score applies nothing while still paying for the
    API calls, so every revisable metric must return a real number."""

    def test_every_revisable_metric_has_a_local_score(self) -> None:
        import candidate_scorer as C
        import controller as CTL

        texts = TH.load(SCORED).scored_texts
        cache, guard = C.ThreadCache(texts), C.GuardCache(texts)
        candidate = "A completely different sentence, typed by somebody else."
        for metric in CTL.REVISABLE:
            base = CTL.local_score(cache, guard, metric, 0)
            swapped = CTL.local_score(cache, guard, metric, 0, candidate)
            self.assertFalse(math.isnan(base), msg=f"{metric} base is NaN")
            self.assertFalse(math.isnan(swapped), msg=f"{metric} candidate is NaN")

    def test_unrevisable_metrics_are_excluded_from_targets(self) -> None:
        """Both are pairwise over the whole thread, so neither can be scored per
        comment. self_bertscore is still fixed -- as a member of the similarity
        group, carried by the two metrics that do have exact per-comment forms.
        The stand-in it used to be optimized against got the direction of the
        official metric right on 21 of 36 swaps, which is noise."""
        import controller as CTL

        self.assertNotIn("hard_disagree_rate", CTL.REVISABLE)
        self.assertNotIn("self_bertscore_mean_f1", CTL.REVISABLE)
        self.assertIn("self_bertscore_mean_f1", S.SIMILARITY)
        self.assertFalse(hasattr(CTL, "bertscore_proxy"))
        for metric in CTL.REVISABLE:
            self.assertIn(metric, J.M12)


class RoundSmokeTest(unittest.TestCase):
    """Exercise the whole round body with a stub reviser, so a name error or a
    misordered binding fails here and not after the API has been paid for.

    Four consecutive rounds were lost to `UnboundLocalError: guard` because the
    binding sat one line below its first use; every one of them had already
    spent its API calls before the exception was raised.
    """

    def test_round_applies_and_gates_without_calling_an_api(self) -> None:
        import controller as CTL
        import reviser as R

        tags = [f"v157_20260903_p{i}" for i in (5, 7, 2, 4)]
        out = Path("/tmp/selfloop_smoke")
        if out.exists():
            shutil.rmtree(out)
        states = CTL.stage(tags, out, force=True)
        if len(states) < 3:
            self.skipTest("cohort not staged")
        for state in states:
            CTL.rescore(state, only=(), device="cpu")

        def fake_propose(api, targets, **kwargs):
            # One candidate per target: the comment with its clauses reversed,
            # which is a real edit and needs no network.
            out = {}
            for t in targets:
                parts = [p.strip() for p in t.text.split(".") if p.strip()]
                body = ". ".join(reversed(parts)) + "." if len(parts) > 1 else t.text + " Honestly."
                out[(t.thread_id, t.index)] = [{"text": body, "what_changed": "reordered"}]
            return out

        original = R.propose
        R.propose = fake_propose
        try:
            for target in ("similarity", "register", "impolite_rate", "length_cv"):
                result = CTL.run_round(states, api=None, model="stub", target=target,
                                       community="Reddit", protected=[], device="cpu",
                                       round_idx=1, workers=1, feedback={}, verbose=False)
                self.assertNotIn("exception", str(result.get("reason", "")), msg=target)
                # Every exit path must report the same shape, or the history
                # file loses fields depending on which branch a round took.
                for key in ("round", "metric", "accepted", "reason", "targets",
                            "applied", "threads_changed", "api_seconds",
                            "score_seconds"):
                    self.assertIn(key, result, msg=f"{target}:{key}")
        finally:
            R.propose = original

    def test_a_rejected_round_leaves_the_previous_text_on_disk(self) -> None:
        """The user's rule: a bad round is not built on. Round 4 continues from
        round 2 when round 3 was rejected -- so a rejected round has to restore
        the text AND the scores AND the file, not just the in-memory copy."""
        import controller as CTL
        import reviser as R

        tags = [f"v157_20260903_p{i}" for i in (5, 7, 2, 4)]
        out = Path("/tmp/selfloop_rollback")
        if out.exists():
            shutil.rmtree(out)
        states = CTL.stage(tags, out, force=True)
        if len(states) < 3:
            self.skipTest("cohort not staged")
        for state in states:
            CTL.rescore(state, only=(), device="cpu")
        before_rows = {s.tag: dict(s.row) for s in states}
        before_disk = {s.tag: (s.work / "discussion.json").read_text() for s in states}

        def fake_propose(api, targets, **kwargs):
            return {(t.thread_id, t.index): [{"text": t.text + " Anyway.",
                                              "what_changed": "appended"}]
                    for t in targets}

        propose, improved = R.propose, J.improved
        R.propose = fake_propose
        J.improved = lambda *a, **k: False      # force every subset to be rejected
        feedback: dict[str, str] = {}
        try:
            result = CTL.run_round(states, api=None, model="stub", target="similarity",
                                   community="Reddit", protected=[], device="cpu",
                                   round_idx=1, workers=1, feedback=feedback,
                                   verbose=False)
        finally:
            R.propose, J.improved = propose, improved
        self.assertFalse(result["accepted"])
        self.assertGreater(result["applied"], 0, "nothing was tried, so nothing was rolled back")
        for state in states:
            self.assertEqual(before_disk[state.tag],
                             (state.work / "discussion.json").read_text(), msg=state.tag)
            self.assertEqual(before_rows[state.tag], state.row, msg=state.tag)
            # and the score files must describe the restored text, or the next
            # round cannot reuse them and pays three transformers to find out.
            import candidate_scorer as C

            self.assertIsNotNone(C.stored_state(state.work, state.thread.scored_texts),
                                 msg=f"{state.tag}: scores left describing a discarded rewrite")
        self.assertTrue(feedback, "a rolled-back rewrite must be remembered")


class DirectionTest(unittest.TestCase):
    def test_direction_follows_the_widest_gap_not_the_first_member(self) -> None:
        """28 of 106 celebrity threads had self_bertscore a hair below real
        while semantic cosine sat far above it. Reading the direction off the
        group's first member told the model to make those threads MORE alike."""
        import controller as CTL

        state = CTL.ThreadState(
            tag="t", work=Path("."), thread=None,
            row={"self_bertscore_mean_f1": 0.4698, "semantic_mean_cosine": 0.2291,
                 "self_bleu_4": 0.0300},
            real={"self_bertscore_mean_f1": 0.4756, "semantic_mean_cosine": 0.1397,
                  "self_bleu_4": 0.0315})
        real = {m: CTL.thread_target(state, m) for m in S.SIMILARITY}
        dominant = CTL.dominant_metric(state, S.SIMILARITY, real)
        self.assertEqual("semantic_mean_cosine", dominant)
        self.assertTrue(float(state.row[dominant]) > real[dominant],
                        "this thread is too self-similar and must be told so")
        self.assertFalse(float(state.row["self_bertscore_mean_f1"])
                         > real["self_bertscore_mean_f1"],
                         "the first member alone would have said the opposite")


class TargetOrderTest(unittest.TestCase):
    def test_similarity_group_comes_before_any_register_metric(self) -> None:
        import controller as CTL

        mk = lambda d, p: J.MetricVerdict("m", 0, 0, p, p, d)
        v = {"self_bertscore_mean_f1": mk(0.21, 0.007),
             "semantic_mean_cosine": mk(0.48, 0.0),
             "self_bleu_4": mk(0.01, 0.9),
             "polite_rate": mk(0.64, 0.0),        # much worse |d|, still second
             "emotion_entropy": mk(0.61, 0.0),
             "length_cv": mk(0.17, 0.035)}
        self.assertEqual("similarity", CTL._next_target(v, set()))
        self.assertEqual("register", CTL._next_target(v, {"similarity"}))
        self.assertEqual("length_cv", CTL._next_target(v, {"similarity", "register"}))

    def test_a_passing_group_is_skipped(self) -> None:
        import controller as CTL

        mk = lambda d, p: J.MetricVerdict("m", 0, 0, p, p, d)
        v = {"self_bertscore_mean_f1": mk(0.02, 0.9),
             "semantic_mean_cosine": mk(0.03, 0.8),
             "self_bleu_4": mk(0.01, 0.9),
             "polite_rate": mk(0.64, 0.0)}
        self.assertEqual("register", CTL._next_target(v, set()))


class BaselineReuseTest(unittest.TestCase):
    """The staged baseline must equal what combined_eval.py reports, or the
    loop starts from different numbers than the project does."""

    def test_staged_rows_reproduce_the_published_verdict(self) -> None:
        import csv
        import controller as CTL

        tags = [f"v157_20260903_p{i}" for i in range(10)]
        out = Path("/tmp/selfloop_baseline")
        if out.exists():
            shutil.rmtree(out)
        states = CTL.stage(tags, out, force=True)
        if len(states) < 3:
            self.skipTest("cohort not staged")
        staged = J.verdict([s.row for s in states], [s.real for s in states])

        gen, real = [], []
        for tag in tags:
            base = REPO / f"artifacts/generalized_card/runs/{tag}/matched_evaluation"
            gen += [r for r in csv.DictReader((base / "matched_generated_thread_scores.csv").open())
                    if not r["thread_id"].startswith("__")]
            real += [r for r in csv.DictReader((base / "matched_real_thread_scores.csv").open())
                     if not r["thread_id"].startswith("__")]
        published = J.verdict(gen, real)
        self.assertEqual(set(published), set(staged))
        for key, item in published.items():
            self.assertAlmostEqual(item.d, staged[key].d, places=6, msg=key)
            self.assertEqual(item.passes, staged[key].passes, msg=key)
