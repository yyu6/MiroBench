from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from .io_utils import read_csv_rows, write_csv_rows, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT / "data" / "stage1" / "product_descriptions.csv"
DEFAULT_LOG_JSONL = ROOT / "data" / "stage1" / "openai_meta_description_runs.jsonl"
PROMPT_VERSION = "meta-description-v1"
REVIEW_MARKER = "NEEDS_HUMAN_REVIEW"

SYSTEM_PROMPT = """You extract official card-page product descriptions for a dataset.

Use only content retrieved from the official product page domain via OpenAI web search.
Do not invent benefits, offers, fees, or terms.
Do not summarize generic issuer navigation, footer text, login text, cookie banners, or legal boilerplate.
Return plain text only with no markdown, no bullets, no labels, and no surrounding quotes.
Never output NEEDS_HUMAN_REVIEW.
If you are uncertain, return the best matching official description text you can find on the official domain.
"""

USER_PROMPT_TEMPLATE = """Card metadata:
- card_id: {card_id}
- card_name: {card_name}
- issuer: {issuer}
- official_product_url: {official_product_url}

Task:
Use OpenAI web search to find and read the official product page for this card and return the single best raw official product description to store in the CSV `meta_description` column.

Requirements:
- Search only within the official product URL domain.
- Prioritize the exact `official_product_url`. If that exact page is unavailable, use the closest official page on the same domain that clearly matches `card_name`.
- Prefer the primary marketing summary or strongest descriptive paragraph for this exact card.
- Keep important product numbers, reward rates, credits, annual-fee wording, and key value propositions when they are part of the official description.
- Exclude navigation, menus, unrelated promos, footer text, account-management text, and disclosure-only fragments.
- If multiple cards or variants appear, choose the text that best matches `card_name`.
- Never return NEEDS_HUMAN_REVIEW.
- Do not include citations, URLs, markdown links, brackets, or source attributions in the output.

some of the description is in the url
 page directly and some are contain in the page that has another hyperlinked text
something like: View important rates and disclosures, Rewards terms, Rates and Fees,
pricing and terms, offer detail(but some leads to exact the same page as the product url,
and some pop up a new small screen with actual more details - chase sephaire - for
example), 1Important Pricing & Information +
2Additional Information +, and these are just example of hyperlinks, some may vary a
little, but give me thorough description for each card.

Output:
Return only the raw description text for `meta_description`.
"""


def _clean_output(value: str) -> str:
    cleaned = " ".join((value or "").split()).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\((?:https?://|www\.)[^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[0-9,\s]+\]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -;,.")
    if cleaned == REVIEW_MARKER:
        return ""
    return cleaned


def _build_prompt(row: dict[str, str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        card_id=row.get("card_id", "").strip(),
        card_name=row.get("card_name", "").strip(),
        issuer=row.get("issuer", "").strip(),
        official_product_url=row.get("official_product_url", "").strip(),
    )


def _allowed_domain(url: str) -> str:
    return (urlparse(url).hostname or "").strip().lower()


def _extract_sources(response: object) -> list[str]:
    try:
        payload = response.model_dump()
    except Exception:
        return []

    urls: list[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        for source in action.get("sources", []) or []:
            url = (source or {}).get("url", "")
            if url:
                urls.append(url)

    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _source_matches_official_domain(sources: list[str], official_url: str) -> bool:
    official_host = _allowed_domain(official_url)
    if not official_host:
        return False
    for source in sources:
        host = _allowed_domain(source)
        if host == official_host or host.endswith(f".{official_host}"):
            return True
    return False


def _request_meta_description(
    client: OpenAI,
    model: str,
    row: dict[str, str],
    max_attempts: int = 3,
) -> tuple[str, str, list[str], bool]:
    prompt = _build_prompt(row)
    url = row.get("official_product_url", "").strip()
    allowed_domain = _allowed_domain(url)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": [allowed_domain]} if allowed_domain else {},
                    }
                ],
                tool_choice="auto",
                include=["web_search_call.action.sources"],
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                ],
            )
            text = _clean_output(response.output_text)
            sources = _extract_sources(response)
            source_match = _source_matches_official_domain(sources, url)
            return text, getattr(response, "id", ""), sources, source_match
        except Exception as exc:  # pragma: no cover - runtime/API path
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"OpenAI request failed for {row.get('card_id', '')}: {last_error}")


def refresh_meta_descriptions_with_openai(
    input_csv: Path,
    output_csv: Path,
    log_jsonl: Path,
    model: str = "gpt-4.1",
    row_limit: int | None = None,
    checkpoint_every: int = 5,
    sleep_seconds: float = 0.0,
    print_prompts: bool = False,
) -> None:
    load_dotenv()
    if print_prompts:
        print("SYSTEM PROMPT")
        print(SYSTEM_PROMPT)
        print("\nUSER PROMPT TEMPLATE")
        print(USER_PROMPT_TEMPLATE)
        print("")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    rows = read_csv_rows(input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {input_csv}.")

    fieldnames = list(rows[0].keys())
    if "meta_description" not in fieldnames:
        raise RuntimeError("Input CSV is missing the `meta_description` column.")

    client = OpenAI()
    run_log: list[dict[str, str]] = []
    processed = 0

    for row in rows:
        url = row.get("official_product_url", "").strip()
        if not url:
            continue
        if row_limit is not None and processed >= row_limit:
            break

        old_value = row.get("meta_description", "")
        card_id = row.get("card_id", "")
        try:
            new_value, response_id, source_urls, source_match = _request_meta_description(client, model, row)
            if not new_value:
                new_value = old_value
                row["meta_description"] = new_value
                status = "fallback_old_value"
            elif source_match:
                row["meta_description"] = new_value
                status = "updated"
            else:
                row["meta_description"] = new_value
                status = "updated_unverified_source"
            error = ""
        except Exception as exc:  # pragma: no cover - runtime/API path
            new_value = old_value
            response_id = ""
            source_urls = []
            source_match = False
            status = "error"
            error = str(exc)

        run_log.append(
            {
                "card_id": card_id,
                "card_name": row.get("card_name", ""),
                "official_product_url": url,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "status": status,
                "old_meta_description": old_value,
                "new_meta_description": new_value,
                "response_id": response_id,
                "fetched_by_llm": status != "error",
                "official_domain_match": source_match,
                "source_urls": source_urls,
                "error": error,
            }
        )
        processed += 1
        preview = (new_value[:140] + "...") if len(new_value) > 143 else new_value
        print(
            f"[{processed}] {card_id} status={status} "
            f"fetched_by_llm={status != 'error'} official_domain_match={source_match} "
            f"sources={len(source_urls)} preview={preview}"
        )

        if processed % checkpoint_every == 0:
            write_csv_rows(output_csv, rows, fieldnames)
            write_jsonl(log_jsonl, run_log)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_csv_rows(output_csv, rows, fieldnames)
    write_jsonl(log_jsonl, run_log)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use the OpenAI Responses API to refresh `meta_description` from official product URLs."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--log-jsonl", type=Path, default=DEFAULT_LOG_JSONL)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--print-prompts", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
