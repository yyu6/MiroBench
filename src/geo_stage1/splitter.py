from __future__ import annotations

import random
from pathlib import Path

from .io_utils import read_csv_rows, write_csv_rows


def select_cards(
    seed_csv: Path,
    product_csv: Path,
    coverage_csv: Path,
    output_csv: Path,
    target_cards: int = 100,
    min_total_items: int = 100,
    min_comment_items: int = 25,
) -> None:
    seed_rows = {row["card_id"]: row for row in read_csv_rows(seed_csv)} if seed_csv.exists() else {}
    product_rows = {row["card_id"]: row for row in read_csv_rows(product_csv)}
    coverage_rows = read_csv_rows(coverage_csv)

    selected_rows: list[dict[str, object]] = []
    for coverage in coverage_rows:
        card_id = coverage["card_id"]
        product = product_rows.get(card_id)
        if not product:
            continue
        if not product["official_product_url"]:
            continue
        if int(coverage["total_items"]) < min_total_items:
            continue
        if int(coverage["comment_count"]) < min_comment_items:
            continue

        seed = seed_rows.get(card_id, {})
        selected_rows.append(
            {
                "card_id": card_id,
                "card_name": product.get("card_name") or seed.get("card_name", ""),
                "issuer": product.get("issuer") or seed.get("issuer", ""),
                "customer_segment": product.get("target_customer_segment") or seed.get("customer_segment", ""),
                "reward_family": product.get("reward_program_guess") or seed.get("reward_family", ""),
                "official_product_url": product["official_product_url"],
                "meta_description": product.get("official_marketing_description") or product.get("meta_description", ""),
                "annual_fee_text": product["annual_fee_text"],
                "submission_count": coverage["submission_count"],
                "comment_count": coverage["comment_count"],
                "total_items": coverage["total_items"],
            }
        )

    selected_rows.sort(key=lambda row: int(row["total_items"]), reverse=True)
    selected_rows = selected_rows[:target_cards]

    write_csv_rows(
        output_csv,
        selected_rows,
        [
            "card_id",
            "card_name",
            "issuer",
            "customer_segment",
            "reward_family",
            "official_product_url",
            "meta_description",
            "annual_fee_text",
            "submission_count",
            "comment_count",
            "total_items",
        ],
    )


def split_cards(selected_csv: Path, output_csv: Path, train_ratio: float = 0.7, seed: int = 17) -> None:
    rows = read_csv_rows(selected_csv)
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    train_cutoff = int(len(rows) * train_ratio)
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        split = "train" if index < train_cutoff else "test"
        output_rows.append(
            {
                "card_id": row["card_id"],
                "card_name": row["card_name"],
                "issuer": row["issuer"],
                "split": split,
            }
        )

    write_csv_rows(output_csv, output_rows, ["card_id", "card_name", "issuer", "split"])
