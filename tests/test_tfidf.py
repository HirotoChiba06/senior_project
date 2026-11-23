"""Fixture-based TF-IDF tests: no project interview text is required."""
import csv
import math

import pytest

from shogi_analysis.tfidf import eligible_term, export_tfidf, load_documents


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def fixture(tmp_path):
    utterances = tmp_path / "utterances.csv"
    tokens = tmp_path / "tokens_merged.csv"
    docs = [
        ("2017", "2017-0001", "player"),
        ("2017", "2017-0002", "player"),
        ("2020", "2020-0001", "player"),
        ("2020", "2020-0002", "question"),
        ("2023", "2023-0001", "player"),
        ("2026", "2026-0001", "player"),
    ]
    write_csv(utterances, ["year", "utterance_id", "role", "speaker", "source_file",
                           "printed_page_start"], [
        {"year": year, "utterance_id": uid, "role": role, "speaker": "棋士" if role == "player" else "質問者",
         "source_file": f"{year}.txt", "printed_page_start": "39"} for year, uid, role in docs
    ])
    terms = {
        "2017-0001": [("ＡＩ", "名詞", "1"), ("研究", "名詞", "1"), ("。", "補助記号", "0")],
        "2017-0002": [("研究", "名詞", "1"), ("する", "動詞", "1")],
        "2020-0001": [("AI", "名詞", "1"), ("値", "名詞", "1")],
        "2020-0002": [("質問", "名詞", "1")],
        "2023-0001": [("理解", "名詞", "1"), ("1", "名詞", "1"), ("一", "名詞", "1")],
        "2026-0001": [("は", "助詞", "0")],
    }
    rows = []
    for year, uid, role in docs:
        for term, pos, is_content in terms[uid]:
            rows.append({"year": year, "utterance_id": uid, "role": role,
                         "normalized_form": term, "pos1": pos, "is_content_word": is_content})
    write_csv(tokens, ["year", "utterance_id", "role", "normalized_form", "pos1", "is_content_word"], rows)
    return utterances, tokens


def test_eligible_terms_filter_and_normalization():
    base = {"is_content_word": "1", "pos1": "名詞"}
    assert eligible_term({**base, "normalized_form": "ＡＩ"}, set()) == "AI"
    assert eligible_term({**base, "normalized_form": "角"}, set()) == "角"  # one-character term retained
    for term in ("", "1", "一", "12.5", "。", "!"):
        assert eligible_term({**base, "normalized_form": term}, set()) is None
    assert eligible_term({**base, "normalized_form": "研究"}, {"研究"}) is None
    assert eligible_term({**base, "pos1": "助詞", "normalized_form": "AI"}, set()) is None


def test_tfidf_year_ranks_are_global_and_frequency_is_separate(tmp_path):
    utterances, tokens = fixture(tmp_path)
    originals = (utterances.read_bytes(), tokens.read_bytes())
    out = tmp_path / "out"
    summary = export_tfidf(utterances, tokens, out, top_n=2)
    assert summary["player_utterances"] == 5
    assert summary["eligible_tokens"] == 7
    assert summary["vocabulary"] == 5  # AI, 研究, する, 値, 理解
    assert summary["years"] == 4
    assert (utterances.read_bytes(), tokens.read_bytes()) == originals

    rows = read_rows(out / "tfidf_year_top_terms.csv")
    assert len([r for r in rows if r["year"] == "2017"]) == 2
    assert not [r for r in rows if r["year"] == "2026"]
    assert {r["term"] for r in rows if r["year"] == "2017"} == {"する", "研究"}
    assert all(r["player_utterances_in_year"] == "2" for r in rows if r["year"] == "2017")
    assert all(r["player_utterances_in_year"] == "1" for r in rows if r["year"] == "2020")

    freq = read_rows(out / "term_frequency.csv")
    research = next(r for r in freq if r["year"] == "2017" and r["term"] == "研究")
    assert research["count"] == "2" and research["utterances_with_term"] == "2"
    assert math.isclose(float(research["per_10000_eligible_tokens"]), 5000.0)
    assert all(r["term"] != "質問" for r in freq)

    tfidf = read_rows(out / "tfidf_utterances.csv")
    assert {r["utterance_id"] for r in tfidf} == {"2017-0001", "2017-0002", "2020-0001", "2023-0001"}
    assert all(r["source_file"].endswith(".txt") and r["printed_page_start"] == "39" for r in tfidf)
    ai_rows = [r for r in tfidf if r["term"] == "AI"]
    assert len(ai_rows) == 2
    assert all(float(r["tfidf"]) > 0 for r in ai_rows)
    # sklearn's smoothed IDF is fitted over all FIVE player utterances, not
    # separately for each year: idf(t) = log((1+N)/(1+df(t))) + 1.
    doc2 = {r["term"]: float(r["tfidf"]) for r in tfidf if r["utterance_id"] == "2017-0002"}
    expected_ratio = (1 + math.log(6 / 2)) / (1 + math.log(6 / 3))
    assert math.isclose(doc2["する"] / doc2["研究"], expected_ratio, rel_tol=1e-8)


def test_stopwords_changes_derived_tables_only(tmp_path):
    utterances, tokens = fixture(tmp_path)
    words = tmp_path / "stopwords.txt"
    words.write_text("# optional\nＡＩ\n研究\n", encoding="utf-8")
    documents = load_documents(utterances, tokens, {"AI", "研究"})
    assert documents[0].terms == ()
    result = export_tfidf(utterances, tokens, tmp_path / "out", stopwords_file=words)
    assert result["vocabulary"] == 3
    assert not any(r["term"] in {"AI", "研究"} for r in read_rows(tmp_path / "out" / "term_frequency.csv"))


def test_invalid_inputs_and_no_terms(tmp_path):
    utterances, tokens = fixture(tmp_path)
    with pytest.raises(ValueError, match="top_n"):
        export_tfidf(utterances, tokens, tmp_path, top_n=0)
    with pytest.raises(ValueError, match="different paths"):
        export_tfidf(tokens, tokens, tmp_path)
    words = tmp_path / "words.txt"
    words.write_text("AI\n研究\nする\n値\n理解\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No eligible terms"):
        export_tfidf(utterances, tokens, tmp_path / "empty", stopwords_file=words)


def test_bad_token_role_and_unknown_id(tmp_path):
    utterances, tokens = fixture(tmp_path)
    rows = read_rows(tokens)
    rows[0]["role"] = "question"
    write_csv(tokens, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Year/role mismatch"):
        load_documents(utterances, tokens, set())
    rows[0]["role"] = "player"
    rows[0]["utterance_id"] = "missing"
    write_csv(tokens, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Unknown utterance_id"):
        load_documents(utterances, tokens, set())


def test_missing_columns(tmp_path):
    utterances, tokens = fixture(tmp_path)
    write_csv(tokens, ["year", "utterance_id"], [])
    with pytest.raises(ValueError, match="Missing columns"):
        load_documents(utterances, tokens, set())
