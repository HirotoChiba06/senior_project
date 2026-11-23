"""Phase 2 Step 1.5: auditable merging of two domain-specific compounds.

The original morpheme CSV is read-only. A separate derived CSV records the
source-token span for every row and never joins terms across utterances or
across whitespace/punctuation in the source text.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .tokenizer import TOKEN_FIELDS

# Explicitly defined before inspecting downstream TF-IDF results.
MERGE_RULES = {
    ("振り", "飛車"): "振り飛車",
    ("評価", "値"): "評価値",
}
EXTRA_FIELDS = ["source_token_start", "source_token_end", "was_merged", "merge_rule"]
OUTPUT_FIELDS = TOKEN_FIELDS + EXTRA_FIELDS


def _align_to_source(tokens: Sequence[Mapping[str, str]], text: str) -> list[tuple[int, int]]:
    """Locate each surface in order; permit whitespace gaps but not missing text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        surface = token["surface"]
        if not surface:
            raise ValueError("Empty token surface cannot be aligned")
        start = text.find(surface, cursor)
        if start < 0 or text[cursor:start].strip():
            raise ValueError(
                f"Cannot align token {surface!r} in utterance {token['utterance_id']} "
                "with the original text"
            )
        end = start + len(surface)
        spans.append((start, end))
        cursor = end
    if text[cursor:].strip():
        raise ValueError("Unaligned non-whitespace text at end of utterance")
    return spans


def merge_utterance(
    tokens: Sequence[Mapping[str, str]], original_text: str
) -> tuple[list[dict[str, str | int]], Counter[str]]:
    """Merge only exact, directly adjacent surface-token pairs within one utterance.

    Input token_index values must enumerate the unmodified Sudachi output.
    The output token_index is consecutive *after* merges; source-token indices
    are retained separately for traceability.
    """
    if not tokens:
        if original_text.strip():
            raise ValueError("Nonempty source text has no tokens")
        return [], Counter()
    ids = {t["utterance_id"] for t in tokens}
    if len(ids) != 1:
        raise ValueError("Tokens from different utterances must not be merged")
    actual = [int(t["token_index"]) for t in tokens]
    if actual != list(range(1, len(tokens) + 1)):
        raise ValueError("Original token_index values must be consecutive from 1")

    spans = _align_to_source(tokens, original_text)
    result: list[dict[str, str | int]] = []
    counts: Counter[str] = Counter()
    i = 0
    while i < len(tokens):
        first = tokens[i]
        matched = False
        if i + 1 < len(tokens):
            second = tokens[i + 1]
            pair = first["surface"], second["surface"]
            term = MERGE_RULES.get(pair)
            if term is not None and spans[i][1] == spans[i + 1][0]:
                # Verify that the original literal text contains precisely the term.
                if original_text[spans[i][0]:spans[i + 1][1]] == term:
                    row = dict(first)
                    row.update({
                        "surface": term,
                        "dictionary_form": term,
                        "normalized_form": term,
                        "pos1": "名詞",
                        "pos2": "普通名詞",
                        "pos3": "一般",
                        "pos4": "*",
                        "is_content_word": 1,
                        "source_token_start": first["token_index"],
                        "source_token_end": second["token_index"],
                        "was_merged": 1,
                        "merge_rule": term,
                        "token_index": len(result) + 1,
                    })
                    result.append(row)
                    counts[term] += 1
                    i += 2
                    matched = True
        if not matched:
            row = dict(first)
            row.update({
                "source_token_start": first["token_index"],
                "source_token_end": first["token_index"],
                "was_merged": 0,
                "merge_rule": "",
                "token_index": len(result) + 1,
            })
            result.append(row)
            i += 1
    return result, counts


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        return list(reader)


def export_merged_tokens(
    tokens_csv: Path, utterances_csv: Path, output_csv: Path
) -> dict[str, int]:
    """Read both Phase 1/2 CSVs and write a distinct, auditable derived file."""
    resolved = {p.resolve() for p in (tokens_csv, utterances_csv, output_csv)}
    if len(resolved) != 3:
        raise ValueError("Input and output paths must all be different")

    utterances = _read_csv(utterances_csv, {"utterance_id", "text"})
    source_texts: dict[str, str] = {}
    for row in utterances:
        uid = row["utterance_id"]
        if uid in source_texts:
            raise ValueError(f"Duplicate utterance_id in source: {uid}")
        source_texts[uid] = row["text"]

    all_tokens = _read_csv(tokens_csv, set(TOKEN_FIELDS))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_tokens:
        uid = row["utterance_id"]
        if uid not in source_texts:
            raise ValueError(f"Unknown utterance_id in tokens: {uid}")
        grouped[uid].append(row)

    merged_rows: list[dict[str, str | int]] = []
    counts: Counter[str] = Counter()
    for uid, text in source_texts.items():
        tokens = grouped.get(uid, [])
        rows, matches = merge_utterance(tokens, text)
        merged_rows.extend(rows)
        counts.update(matches)

    # Do not touch the original tokens.csv, even if a validation step fails.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    return {
        "utterances": len(source_texts),
        "original_tokens": len(all_tokens),
        "merged_tokens": len(merged_rows),
        "振り飛車": counts["振り飛車"],
        "評価値": counts["評価値"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two explicitly chosen shogi compound terms")
    parser.add_argument("--tokens", type=Path, default=Path("outputs/tables/tokens.csv"))
    parser.add_argument("--utterances", type=Path, default=Path("outputs/tables/utterances.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/tokens_merged.csv"))
    args = parser.parse_args()
    result = export_merged_tokens(args.tokens, args.utterances, args.output)
    print("Domain-term merging:", result)
    print("CSV output:", args.output)


if __name__ == "__main__":
    main()
