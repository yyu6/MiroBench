"""Render leaderboard.json into the two table bodies of docs/leaderboard.html.

The HTML carries marker comments:

    <!-- LB:summary START -->   ... summary-by-family <tr> rows ...   <!-- LB:summary END -->
    <!-- LB:perdomain START --> ... per-domain <tr> rows ...         <!-- LB:perdomain END -->

render() replaces the content between each pair, leaving everything else
(styling, prose, scaffolding) untouched.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .families import (
    DOMAIN_ORDER,
    FAMILY_ORDER,
    METRICS_PER_DOMAIN,
    heat_class,
    total_style,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "docs" / "leaderboard.json"
HTML_PATH = REPO_ROOT / "docs" / "leaderboard.html"

IND = "          "  # 10-space indent, matching the existing rows.


def _fmt(x: float, places: int) -> str:
    q = Decimal(10) ** -places
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def fmt3(x: float) -> str:
    return _fmt(x, 3)


def fmt_ratio(value: float, base: float) -> str:
    if not base:
        return "—"
    return _fmt(value / base, 1)


def _total(entry: dict[str, Any]) -> tuple[int, int]:
    passes = sum(entry["domain_pass"].values())
    denom = METRICS_PER_DOMAIN * len(entry["domains"])
    return passes, denom


def _sorted_entries(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: (-sum(e["domain_pass"].values()), e["model"]))


# ----------------------------- summary table ------------------------------- #

def _summary_baseline_row(baseline: dict) -> str:
    fams = baseline["families"]
    cells = []
    for fam in FAMILY_ORDER:
        b = fams[fam]
        cells.append(
            f'<td class="text-right text-mute"><div>W₁ '
            f'<span class="text-text">{fmt3(b["w1"])}</span></div>'
            f'<div class="text-[12px]">|δ| '
            f'<span class="text-text">{fmt3(b["cliffs"])}</span></div></td>'
        )
    return (
        f'{IND}<tr style="background:#F7F7F4;">\n'
        f'{IND}  <td class="text-mute2">—</td>\n'
        f'{IND}  <td>\n'
        f'{IND}    <div class="font-sans text-mute text-[12.5px]">{baseline["label"]}</div>\n'
        f'{IND}    <div class="font-mono text-[11px] text-mute2">{baseline["sublabel"]}</div>\n'
        f'{IND}  </td>\n'
        f'{IND}  {"".join(cells)}\n'
        f'{IND}  <td class="text-right text-mute2" style="border-left:1px solid #E4E4E0;">— baseline —</td>\n'
        f'{IND}</tr>'
    )


def _summary_entry_row(rank: int, entry: dict, baseline: dict) -> str:
    fams = entry["families"]
    bfam = baseline["families"]
    cells = []
    for fam in FAMILY_ORDER:
        fv = fams.get(fam)
        if not fv:
            cells.append('<td class="text-right text-mute2">—</td>')
            continue
        r1 = fmt_ratio(fv["w1"], bfam[fam]["w1"])
        r2 = fmt_ratio(fv["cliffs"], bfam[fam]["cliffs"])
        cells.append(
            f'<td class="text-right"><div>W₁ {fmt3(fv["w1"])} '
            f'<span class="text-mute2 text-[11px]">×{r1}</span></div>'
            f'<div class="text-[12px] text-mute">|δ| {fmt3(fv["cliffs"])} '
            f'<span class="text-mute2 text-[11px]">×{r2}</span></div></td>'
        )
    passes, denom = _total(entry)
    pct = passes / denom * 100 if denom else 0.0
    cls, fw = total_style(pct)
    return (
        f'{IND}<tr>\n'
        f'{IND}  <td>{rank}</td>\n'
        f'{IND}  <td><div class="font-sans text-text">{entry["engine"]}</div>'
        f'<div class="font-mono text-[11.5px] text-mute2">{entry["model"]}</div></td>\n'
        f'{IND}  {"".join(cells)}\n'
        f'{IND}  <td class="text-right" style="border-left:1px solid #E4E4E0;">'
        f'<span class="{cls} text-[16px]{fw}">{passes} / {denom}</span>'
        f'<div class="text-mute2 text-[12px]">{_fmt(pct, 1)}%</div></td>\n'
        f'{IND}</tr>'
    )


def _pending_summary_row(p: dict) -> str:
    dash = '<td class="text-right">—</td>'
    return (
        f'{IND}<tr class="text-mute2">\n'
        f'{IND}  <td>—</td>\n'
        f'{IND}  <td><div class="font-sans">{p["engine"]}</div>'
        f'<div class="font-mono text-[11px]">{p["model"]} · pending</div></td>\n'
        f'{IND}  {dash * 5}\n'
        f'{IND}  <td class="text-right" style="border-left:1px solid #E4E4E0;">—</td>\n'
        f'{IND}</tr>'
    )


def render_summary(data: dict) -> str:
    baseline = data["baseline"]
    full = [e for e in _sorted_entries(data["entries"]) if len(e["domains"]) == len(DOMAIN_ORDER)]
    rows = [_summary_baseline_row(baseline)]
    for i, e in enumerate(full, start=1):
        rows.append(_summary_entry_row(i, e, baseline))
    for p in data.get("pending", []):
        rows.append(_pending_summary_row(p))
    return "\n".join(rows)


# --------------------------- per-domain table ------------------------------ #

def _model_cell(entry: dict) -> str:
    base = (f'<div class="font-sans text-text">{entry["engine"]}</div>'
            f'<div class="font-mono text-[11.5px] text-mute2">{entry["model"]}</div>')
    if entry.get("tier") == "community":
        ns = entry.get("domain_n", {})
        n = max(ns.values()) if ns else entry.get("n_threads", "?")
        base += f'<div class="font-mono text-[10.5px] text-mute2">community · n={n}</div>'
    return f'<td>{base}</td>'


def _perdomain_entry_row(entry: dict) -> str:
    cells = []
    for d in DOMAIN_ORDER:
        if d in entry["domain_pass"]:
            c = entry["domain_pass"][d]
            cells.append(f'<td class="text-right"><span class="{heat_class(c)}">'
                         f'{c}/{METRICS_PER_DOMAIN}</span></td>')
        else:
            cells.append('<td class="text-right text-mute2">—</td>')
    passes, denom = _total(entry)
    pct = passes / denom * 100 if denom else 0.0
    cls, fw = total_style(pct)
    total_cell = (
        f'<td class="text-right" style="border-left:1px solid #E4E4E0;">'
        f'<span class="{cls}{fw} text-[15px]">{passes}/{denom}</span>'
        f'<div class="text-mute2 text-[11px]">{_fmt(pct, 1)}%</div></td>'
    )
    return (
        f'{IND}<tr>\n'
        f'{IND}  {_model_cell(entry)}\n'
        f'{IND}  {"".join(cells)}\n'
        f'{IND}  {total_cell}\n'
        f'{IND}</tr>'
    )


def _pending_perdomain_row(p: dict) -> str:
    dash = '<td class="text-right">—</td>'
    return (
        f'{IND}<tr class="text-mute2">\n'
        f'{IND}  <td><div class="font-sans">{p["engine"]}</div>'
        f'<div class="font-mono text-[11px]">{p["model"]} · pending</div></td>\n'
        f'{IND}  {dash * len(DOMAIN_ORDER)}\n'
        f'{IND}  <td class="text-right" style="border-left:1px solid #E4E4E0;">—</td>\n'
        f'{IND}</tr>'
    )


def render_perdomain(data: dict) -> str:
    rows = []
    for e in _sorted_entries(data["entries"]):
        rows.append(_perdomain_entry_row(e))
    for p in data.get("pending", []):
        rows.append(_pending_perdomain_row(p))
    return "\n".join(rows)


# ------------------------------- splicing ---------------------------------- #

def _splice(html: str, name: str, body: str) -> str:
    start = f"<!-- LB:{name} START -->"
    end = f"<!-- LB:{name} END -->"
    i, j = html.find(start), html.find(end)
    if i == -1 or j == -1:
        raise ValueError(f"markers for '{name}' not found in HTML "
                         f"(expected {start} ... {end})")
    return html[: i + len(start)] + "\n" + body + "\n" + IND[:-2] + html[j:]


def render(json_path: Path = JSON_PATH, html_path: Path = HTML_PATH) -> str:
    data = json.loads(json_path.read_text())
    html = html_path.read_text()
    html = _splice(html, "summary", render_summary(data))
    html = _splice(html, "perdomain", render_perdomain(data))
    return html


def main(argv: list[str] | None = None) -> int:
    html = render()
    HTML_PATH.write_text(html)
    print(f"Rendered {HTML_PATH}")
    return 0
