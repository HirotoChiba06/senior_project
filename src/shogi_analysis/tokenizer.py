"""Phase 2 / Step 1: morphological analysis of Phase 1 utterance CSV.

No original text is rewritten. Every morpheme is exported, including particles,
auxiliaries and symbols, so subsequent analyses can make their own filter choices.
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Mapping

# Content-word status is advisory, not a filtering operation.
CONTENT_POS = frozenset({"名詞", "動詞", "形容詞"})
REQUIRED_INPUT_FIELDS = frozenset({"utterance_id", "year", "role", "speaker", "text"})
TOKEN_FIELDS = [
    "year", "utterance_id", "role", "speaker", "source_file", "printed_page_start",
    "token_index", "surface", "dictionary_form", "normalized_form",
    "pos1", "pos2", "pos3", "pos4", "is_content_word",
]


def load_sudachi() -> tuple[Any, Any]:
    """Import the optional analyzer only when running Phase 2."""
    try:
        from sudachipy import dictionary, tokenizer as sudachi_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            'SudachiPy is required. Install with: python -m pip install -e ".[nlp]"'
        ) from exc
    return dictionary.Dictionary().create(), sudachi_tokenizer.Tokenizer.SplitMode


def tokenize_utterance(
    utterance: Mapping[str, str], analyzer: Any, mode: Any
) -> Iterator[dict[str, str | int]]:
    """Yield morphemes in positional order with traceable utterance metadata."""
    for index, morpheme in enumerate(analyzer.tokenize(utterance["text"], mode), 1):
        pos = morpheme.part_of_speech()
        normalized_word = unicodedata.normalize("NFKC", morpheme.normalized_form())
        yield {
            "year": utterance["year"],
            "utterance_id": utterance["utterance_id"],
            "role": utterance["role"],
            "speaker": utterance["speaker"],
            "source_file": utterance.get("source_file", ""),
            "printed_page_start": utterance.get("printed_page_start", ""),
            "token_index": index,
            "surface": morpheme.surface(),
            "dictionary_form": morpheme.dictionary_form(),
            "normalized_form": normalized_word,
            "pos1": pos[0],
            "pos2": pos[1],
            "pos3": pos[2],
            "pos4": pos[3],
            "is_content_word": int(pos[0] in CONTENT_POS),
        }


def export_tokens(
    input_csv: Path, output_csv: Path, analyzer: Any, mode: Any
) -> dict[str, int]:
    """Export all tokens from Phase 1 CSV without changing Phase 1's output."""
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("Input and output paths must be different")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    utterance_count = token_count = 0
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_INPUT_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        with output_csv.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=TOKEN_FIELDS)
            writer.writeheader()
            for utterance in reader:
                utterance_count += 1
                for token in tokenize_utterance(utterance, analyzer, mode):
                    writer.writerow(token)
                    token_count += 1
    return {"utterances": utterance_count, "tokens": token_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize Phase 1 utterances using SudachiPy")
    parser.add_argument("--input", type=Path, default=Path("outputs/tables/utterances.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/tokens.csv"))
    parser.add_argument("--mode", choices=("A", "B", "C"), default="C")
    args = parser.parse_args()
    analyzer, split_modes = load_sudachi()
    counts = export_tokens(args.input, args.output, analyzer, getattr(split_modes, args.mode))
    print(f"SudachiPy mode {args.mode}: {counts['utterances']} utterances, {counts['tokens']} tokens")
    print(f"CSV output: {args.output}")


if __name__ == "__main__":
    main()
