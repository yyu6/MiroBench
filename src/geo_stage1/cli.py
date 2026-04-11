from __future__ import annotations

import argparse
from pathlib import Path

from .editorial_enricher import enrich_editorial_descriptions
from .reddit_collector import collect_reddit_items
from .splitter import select_cards, split_cards


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_CSV = ROOT / "data" / "reference" / "card_universe_seed.csv"
DEFAULT_PRODUCT_CSV = ROOT / "data" / "stage1" / "product_descriptions.csv"
DEFAULT_META_DESCRIPTION_LOG = ROOT / "data" / "stage1" / "openai_meta_description_runs.jsonl"
DEFAULT_REDDIT_JSONL = ROOT / "data" / "stage1" / "reddit_items.jsonl"
DEFAULT_COVERAGE_CSV = ROOT / "data" / "stage1" / "reddit_coverage.csv"
DEFAULT_SELECTED_CSV = ROOT / "data" / "stage1" / "selected_cards.csv"
DEFAULT_SPLITS_CSV = ROOT / "data" / "stage1" / "card_splits.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 GEO data collection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    meta = subparsers.add_parser("refresh-meta-descriptions-openai")
    meta.add_argument("--input-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    meta.add_argument("--output-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    meta.add_argument("--log-jsonl", type=Path, default=DEFAULT_META_DESCRIPTION_LOG)
    meta.add_argument("--model", default="gpt-4.1")
    meta.add_argument("--row-limit", type=int)
    meta.add_argument("--checkpoint-every", type=int, default=5)
    meta.add_argument("--sleep-seconds", type=float, default=0.0)
    meta.add_argument("--print-prompts", action="store_true")

    editorial = subparsers.add_parser("enrich-editorial")
    editorial.add_argument("--input-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    editorial.add_argument("--output-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    editorial.add_argument("--checkpoint-every", type=int, default=10)
    editorial.add_argument("--row-limit", type=int)
    editorial.add_argument("--workers", type=int, default=4)

    reddit = subparsers.add_parser("collect-reddit")
    reddit.add_argument("--seed-csv", type=Path, default=DEFAULT_SEED_CSV)
    reddit.add_argument("--output-jsonl", type=Path, default=DEFAULT_REDDIT_JSONL)
    reddit.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV)
    reddit.add_argument("--per-card-target", type=int, default=100)
    reddit.add_argument("--max-submissions-per-query", type=int, default=30)

    selected = subparsers.add_parser("select-cards")
    selected.add_argument("--seed-csv", type=Path, default=DEFAULT_SEED_CSV)
    selected.add_argument("--product-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    selected.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV)
    selected.add_argument("--output-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    selected.add_argument("--target-cards", type=int, default=100)
    selected.add_argument("--min-total-items", type=int, default=100)
    selected.add_argument("--min-comment-items", type=int, default=25)

    split = subparsers.add_parser("split-cards")
    split.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    split.add_argument("--output-csv", type=Path, default=DEFAULT_SPLITS_CSV)
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--seed", type=int, default=17)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "refresh-meta-descriptions-openai":
        from .openai_meta_description_updater import refresh_meta_descriptions_with_openai

        refresh_meta_descriptions_with_openai(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            log_jsonl=args.log_jsonl,
            model=args.model,
            row_limit=args.row_limit,
            checkpoint_every=args.checkpoint_every,
            sleep_seconds=args.sleep_seconds,
            print_prompts=args.print_prompts,
        )
        return

    if args.command == "enrich-editorial":
        enrich_editorial_descriptions(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            checkpoint_every=args.checkpoint_every,
            row_limit=args.row_limit,
            workers=args.workers,
        )
        return

    if args.command == "collect-reddit":
        collect_reddit_items(
            seed_csv=args.seed_csv,
            output_jsonl=args.output_jsonl,
            coverage_csv=args.coverage_csv,
            per_card_target=args.per_card_target,
            max_submissions_per_query=args.max_submissions_per_query,
        )
        return

    if args.command == "select-cards":
        select_cards(
            seed_csv=args.seed_csv,
            product_csv=args.product_csv,
            coverage_csv=args.coverage_csv,
            output_csv=args.output_csv,
            target_cards=args.target_cards,
            min_total_items=args.min_total_items,
            min_comment_items=args.min_comment_items,
        )
        return

    if args.command == "split-cards":
        split_cards(
            selected_csv=args.selected_csv,
            output_csv=args.output_csv,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
