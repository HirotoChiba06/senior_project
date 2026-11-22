"""Parse page-annotated Japanese interviews and count explicit AI-related terms.

Only the standard library is required. The original source files are never modified.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

YEARS = (2017, 2020, 2023, 2026)
FILE_TEMPLATE = "shogi_yearbook_{year}_fulltext.txt"
AI_PATTERN = re.compile(r"(?<![A-Za-z0-9])AI(?:\s*[（(]\s*人工知能\s*[）)])?(?![A-Za-z0-9])|人工知能|ソフト|コンピューター?", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"【(?:([^／】]+\.pdf)／)?PDF\s*(\d+)ページ目(?:／[^】]*)?／誌面([^】]+?)ページ】")
SPEAKER_PATTERN = re.compile(r"^(先崎・木村|先崎|木村|杉本|増田)「(.*)$")
QUESTION_PATTERN = re.compile(r"^―+(.*)$")
SECTION_PATTERN = re.compile(r"^【([^】]+)】$")


def normalized(text: str) -> str:
    """Normalize a *copy* for matching; never overwrite source quotations."""
    return unicodedata.normalize("NFKC", text)


def keyword_mentions(text: str) -> list[dict[str, str | int]]:
    """One mention per AI（人工知能）, even if both forms appear inside it."""
    clean = normalized(text)
    mentions = []
    for match in AI_PATTERN.finditer(clean):
        word = match.group()
        kind = ("AI" if word.upper().startswith("AI") or word == "人工知能"
                else "ソフト" if word.startswith("ソフト") else "コンピュータ")
        mentions.append({"term": kind, "surface": word, "start": match.start(), "end": match.end()})
    return mentions


def parse_interview(path: Path, year: int) -> list[dict]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    utterances: list[dict] = []
    current: dict | None = None
    section = ""
    printed_page = ""
    pdf_page = ""
    pdf_file = f"lbo_{year}.pdf"
    last_question_id = ""

    def finish() -> None:
        nonlocal current, last_question_id
        if current is None:
            return
        # Keep the body wording, but remove the outer quotation mark added by the layout.
        current["text"] = "".join(current.pop("fragments")).removesuffix("」")
        if current["text"]:
            current["characters"] = len(normalized(current["text"]))
            current["mentions"] = keyword_mentions(current["text"])
            current["mention_count"] = len(current["mentions"])
            current["review_needed"] = "[要確認" in current["text"]
            utterances.append(current)
            if current["role"] == "question":
                last_question_id = current["utterance_id"]
        current = None

    for line_no, original_line in enumerate(lines, 1):
        line = original_line.strip()
        if not line:
            continue
        page = PAGE_PATTERN.fullmatch(line)
        if page:
            if page.group(1):
                pdf_file = page.group(1)
            pdf_page = page.group(2)
            printed_page = page.group(3)
            continue
        sec = SECTION_PATTERN.fullmatch(line)
        if sec:
            section = sec.group(1)
            continue
        if line.startswith("---") or line.startswith("===") or line.startswith("[広告]"):
            continue
        if section != "本文":
            continue
        role = speaker = body = None
        question = QUESTION_PATTERN.match(line)
        explicit = SPEAKER_PATTERN.match(line)
        if question:
            role, speaker, body = "question", "質問者", question.group(1)
        elif explicit:
            role, speaker, body = "player", explicit.group(1), explicit.group(2)
        elif line.startswith("「"):
            role, speaker, body = "player", {2017: "不明", 2020: "木村", 2023: "杉本", 2026: "増田"}[year], line[1:]

        if role is not None:
            finish()
            unit_id = f"{year}-{len(utterances) + 1:04d}"
            current = {
                "year": year, "utterance_id": unit_id, "role": role, "speaker": speaker,
                "source_file": path.name, "pdf_file": pdf_file, "pdf_page_start": pdf_page,
                "printed_page_start": printed_page, "printed_page_end": printed_page,
                "line_start": line_no, "line_end": line_no,
                "question_id": last_question_id if role == "player" else "",
                "fragments": [body],
            }
        elif current is not None:
            current["fragments"].append(line)
            current["line_end"] = line_no
            current["printed_page_end"] = printed_page
    finish()
    return utterances


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir: Path, output_dir: Path) -> dict[int, int]:
    units = []
    for year in YEARS:
        source = data_dir / FILE_TEMPLATE.format(year=year)
        if not source.is_file():
            raise FileNotFoundError(f"Missing source: {source}")
        units.extend(parse_interview(source, year))

    fields = ["year", "utterance_id", "role", "speaker", "question_id", "text",
              "characters", "mention_count", "source_file", "pdf_file", "pdf_page_start",
              "printed_page_start", "printed_page_end", "line_start", "line_end", "review_needed"]
    write_csv(output_dir / "utterances.csv", units, fields)

    rows = []
    for year in YEARS:
        for role in ("all", "player", "question"):
            selected = [u for u in units if u["year"] == year and (role == "all" or u["role"] == role)]
            counts = Counter(m["term"] for u in selected for m in u["mentions"])
            chars = sum(u["characters"] for u in selected)
            total = sum(counts.values())
            rows.append({"year": year, "role": role, "utterances": len(selected),
                         "characters": chars, "total_mentions": total,
                         "per_10000_chars": round(total * 10000 / chars, 4) if chars else 0,
                         "AI": counts["AI"], "ソフト": counts["ソフト"],
                         "コンピュータ": counts["コンピュータ"]})
    write_csv(output_dir / "frequency_by_year.csv", rows,
              ["year", "role", "utterances", "characters", "total_mentions",
               "per_10000_chars", "AI", "ソフト", "コンピュータ"])

    kwic_rows = []
    for u in units:
        normalized_text = normalized(u["text"])
        for m in u["mentions"]:
            a, b = int(m["start"]), int(m["end"])
            kwic_rows.append({"year": u["year"], "utterance_id": u["utterance_id"],
                              "role": u["role"], "speaker": u["speaker"],
                              "term": m["term"], "surface": m["surface"],
                              "left_context": normalized_text[max(0, a - 35):a],
                              "right_context": normalized_text[b:b + 35],
                              "source_file": u["source_file"], "line_start": u["line_start"],
                              "printed_page_start": u["printed_page_start"]})
    write_csv(output_dir / "kwic.csv", kwic_rows,
              ["year", "utterance_id", "role", "speaker", "term", "surface",
               "left_context", "right_context", "source_file", "line_start", "printed_page_start"])
    # A response may omit the word 'AI' while answering an AI-related question.
    by_id = {u["utterance_id"]: u for u in units}
    review = [dict(u, reason="reply_to_ai_question") for u in units
              if u["role"] == "player" and not u["mentions"] and u["question_id"]
              and by_id[u["question_id"]]["mentions"]]
    write_csv(output_dir / "context_review_candidates.csv", review, fields + ["reason"])
    return {year: sum(u["year"] == year for u in units) for year in YEARS}
