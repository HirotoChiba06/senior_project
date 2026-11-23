"""Phase 2 / Step 2: utterance-level TF-IDF and year-wise frequency comparison.

The IDF is fitted once to all player utterances; year means include zeros for
utterances without a given term. Input files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

CONTENT_POS = frozenset({"名詞", "動詞", "形容詞"})
OUTPUT_NAMES = ("tfidf_utterances.csv", "tfidf_year_top_terms.csv", "term_frequency.csv")
UTTERANCE_FIELDS = {"year", "utterance_id", "role", "speaker", "source_file", "printed_page_start"}
TOKEN_FIELDS = {"year", "utterance_id", "role", "normalized_form", "pos1", "is_content_word"}


@dataclass(frozen=True)
class Document:
    year: str
    utterance_id: str
    speaker: str
    source_file: str
    printed_page_start: str
    terms: tuple[str, ...]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        return list(reader)


def load_stopwords(path: Path | None) -> set[str]:
    """One normalized term per line; no automatic stopword list is imposed."""
    if path is None:
        return set()
    return {
        unicodedata.normalize("NFKC", line.strip())
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def eligible_term(row: dict[str, str], stopwords: set[str]) -> str | None:
    if row["is_content_word"] != "1" or row["pos1"] not in CONTENT_POS:
        return None
    term = unicodedata.normalize("NFKC", row["normalized_form"]).strip()
    if not term or term in stopwords:
        return None
    if term.isnumeric() or re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", term):
        return None
    if all(unicodedata.category(char)[0] in ("P", "S") for char in term):
        return None
    return term


def load_documents(
    utterances_csv: Path, tokens_csv: Path, stopwords: set[str]
) -> list[Document]:
    utterances = read_csv(utterances_csv, UTTERANCE_FIELDS)
    source_by_id: dict[str, dict[str, str]] = {}
    for row in utterances:
        uid = row["utterance_id"]
        if not uid or uid in source_by_id:
            raise ValueError(f"Missing or duplicated utterance_id: {uid!r}")
        source_by_id[uid] = row

    terms_by_id: dict[str, list[str]] = defaultdict(list)
    for row in read_csv(tokens_csv, TOKEN_FIELDS):
        uid = row["utterance_id"]
        if uid not in source_by_id:
            raise ValueError(f"Unknown utterance_id in tokens: {uid}")
        source = source_by_id[uid]
        if row["year"] != source["year"] or row["role"] != source["role"]:
            raise ValueError(f"Year/role mismatch in tokens for {uid}")
        if source["role"] != "player":
            continue
        term = eligible_term(row, stopwords)
        if term is not None:
            terms_by_id[uid].append(term)

    documents = [
        Document(
            year=row["year"], utterance_id=row["utterance_id"],
            speaker=row["speaker"], source_file=row["source_file"],
            printed_page_start=row["printed_page_start"],
            terms=tuple(terms_by_id[row["utterance_id"]]),
        )
        for row in utterances if row["role"] == "player"
    ]
    if not documents:
        raise ValueError("No player utterances found")
    if not any(doc.terms for doc in documents):
        raise ValueError("No eligible terms in player utterances; check POS and stopwords")
    return documents


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_tfidf(
    utterances_csv: Path, tokens_csv: Path, output_dir: Path,
    top_n: int = 20, stopwords_file: Path | None = None,
) -> dict[str, int]:
    """Export three tables. A shared global IDF allows cross-year comparison."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    inputs = {utterances_csv.resolve(), tokens_csv.resolve()}
    if len(inputs) != 2 or any((output_dir / name).resolve() in inputs for name in OUTPUT_NAMES):
        raise ValueError("Input CSVs and output CSVs must be different paths")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError('Install scikit-learn with: python -m pip install -e ".[analysis]"') from exc

    stopwords = load_stopwords(stopwords_file)
    documents = load_documents(utterances_csv, tokens_csv, stopwords)
    # A callable analyzer preserves already-tokenized Japanese single-character words.
    vectorizer = TfidfVectorizer(analyzer=lambda terms: terms, norm="l2", smooth_idf=True)
    matrix = vectorizer.fit_transform([doc.terms for doc in documents])
    terms = vectorizer.get_feature_names_out().tolist()

    # One output row for every non-zero document/term pair; no source sentences exported.
    doc_rows: list[dict] = []
    year_indices: dict[str, list[int]] = defaultdict(list)
    year_counts: dict[str, Counter[str]] = defaultdict(Counter)
    year_doc_freq: dict[str, Counter[str]] = defaultdict(Counter)
    year_lengths: Counter[str] = Counter()
    for i, doc in enumerate(documents):
        year_indices[doc.year].append(i)
        counts = Counter(doc.terms)
        year_counts[doc.year].update(counts)
        year_doc_freq[doc.year].update(counts.keys())
        year_lengths[doc.year] += len(doc.terms)
        row = matrix.getrow(i)
        pairs = sorted(
            ((terms[idx], float(score)) for idx, score in zip(row.indices, row.data)),
            key=lambda item: (-item[1], item[0]),
        )
        for rank, (term, score) in enumerate(pairs, 1):
            doc_rows.append({
                "year": doc.year, "utterance_id": doc.utterance_id,
                "speaker": doc.speaker, "source_file": doc.source_file,
                "printed_page_start": doc.printed_page_start,
                "rank_in_utterance": rank, "term": term,
                "tfidf": f"{score:.10f}", "term_count": counts[term],
                "eligible_tokens_in_utterance": len(doc.terms),
            })

    term_freq_rows: list[dict] = []
    year_tfidf_rows: list[dict] = []
    for year in sorted(year_indices):
        total_tokens = year_lengths[year]
        num_docs = len(year_indices[year])
        freq_sorted = sorted(year_counts[year].items(), key=lambda item: (-item[1], item[0]))
        frequency_rank = {term: rank for rank, (term, _) in enumerate(freq_sorted, 1)}
        for rank, (term, count) in enumerate(freq_sorted, 1):
            term_freq_rows.append({
                "year": year, "rank": rank, "term": term, "count": count,
                "per_10000_eligible_tokens": f"{count * 10000 / total_tokens:.6f}",
                "utterances_with_term": year_doc_freq[year][term],
                "player_utterances_in_year": num_docs,
                "eligible_tokens_in_year": total_tokens,
            })

        # Mean across ALL player utterances in the year (including zero scores).
        means = matrix[year_indices[year], :].mean(axis=0)
        mean_scores = [float(means[0, idx]) for idx in range(len(terms))]
        best = sorted(
            ((term, mean_scores[idx]) for idx, term in enumerate(terms) if mean_scores[idx] > 0),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
        for rank, (term, score) in enumerate(best, 1):
            count = year_counts[year][term]
            year_tfidf_rows.append({
                "year": year, "rank": rank, "term": term,
                "mean_tfidf": f"{score:.10f}", "frequency_rank": frequency_rank[term],
                "count": count,
                "per_10000_eligible_tokens": f"{count * 10000 / total_tokens:.6f}",
                "utterances_with_term": year_doc_freq[year][term],
                "player_utterances_in_year": num_docs,
                "eligible_tokens_in_year": total_tokens,
            })

    write_csv(output_dir / OUTPUT_NAMES[0], [
        "year", "utterance_id", "speaker", "source_file", "printed_page_start",
        "rank_in_utterance", "term", "tfidf", "term_count", "eligible_tokens_in_utterance",
    ], doc_rows)
    write_csv(output_dir / OUTPUT_NAMES[1], [
        "year", "rank", "term", "mean_tfidf", "frequency_rank", "count",
        "per_10000_eligible_tokens", "utterances_with_term", "player_utterances_in_year",
        "eligible_tokens_in_year",
    ], year_tfidf_rows)
    write_csv(output_dir / OUTPUT_NAMES[2], [
        "year", "rank", "term", "count", "per_10000_eligible_tokens",
        "utterances_with_term", "player_utterances_in_year", "eligible_tokens_in_year",
    ], term_freq_rows)
    return {
        "player_utterances": len(documents),
        "eligible_tokens": sum(len(doc.terms) for doc in documents),
        "vocabulary": len(terms),
        "years": len(year_indices),
        "utterance_term_rows": len(doc_rows),
        "year_top_term_rows": len(year_tfidf_rows),
        "frequency_rows": len(term_freq_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Player-utterance TF-IDF and year-level term frequency")
    parser.add_argument("--utterances", type=Path, default=Path("outputs/tables/utterances.csv"))
    parser.add_argument("--tokens", type=Path, default=Path("outputs/tables/tokens_merged.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--stopwords", type=Path, default=None)
    args = parser.parse_args()
    counts = export_tfidf(args.utterances, args.tokens, args.output_dir,
                          top_n=args.top_n, stopwords_file=args.stopwords)
    print("TF-IDF analysis:", counts)
    for name in OUTPUT_NAMES:
        print("CSV output:", args.output_dir / name)


if __name__ == "__main__":
    main()
