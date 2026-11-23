"""Step 1.5 tests: keep originals, merge exact contiguous terms only."""
import csv

import pytest

from shogi_analysis.term_merger import export_merged_tokens, merge_utterance
from shogi_analysis.tokenizer import TOKEN_FIELDS


def token(surface, idx, uid="2023-0001"):
    return dict(
        year="2023", utterance_id=uid, role="player", speaker="杉本",
        source_file="2023.txt", printed_page_start="39",
        token_index=str(idx), surface=surface, dictionary_form=surface,
        normalized_form=surface, pos1="名詞", pos2="普通名詞", pos3="一般",
        pos4="*", is_content_word="1",
    )


def test_two_compounds_merge_and_original_positions_preserved():
    before = [token("振り", 1), token("飛車", 2), token("と", 3),
              token("評価", 4), token("値", 5), token("。", 6)]
    expected_snapshot = [x.copy() for x in before]
    after, counts = merge_utterance(before, "振り飛車と評価値。")
    assert [x["surface"] for x in after] == ["振り飛車", "と", "評価値", "。"]
    assert [x["token_index"] for x in after] == [1, 2, 3, 4]
    assert [(x["source_token_start"], x["source_token_end"]) for x in after] == [
        ("1", "2"), ("3", "3"), ("4", "5"), ("6", "6")
    ]
    assert after[0]["merge_rule"] == "振り飛車"
    assert after[2]["normalized_form"] == "評価値"
    assert after[2]["was_merged"] == 1
    assert counts == {"振り飛車": 1, "評価値": 1}
    assert before == expected_snapshot


def test_whitespace_blocks_merge_and_existing_word_is_kept():
    group = [token("振り", 1), token("飛車", 2), token("角換わり", 3)]
    after, counts = merge_utterance(group, "振り 飛車角換わり")
    assert [r["surface"] for r in after] == ["振り", "飛車", "角換わり"]
    assert not counts
    assert all(r["was_merged"] == 0 for r in after)


def test_unaligned_text_and_cross_utterance_rejected():
    with pytest.raises(ValueError, match="different utterances"):
        merge_utterance([token("振り", 1), token("飛車", 2, "2023-0002")], "振り飛車")
    with pytest.raises(ValueError, match="Cannot align"):
        merge_utterance([token("振り", 1), token("飛車", 2)], "振り、飛車")
    with pytest.raises(ValueError, match="consecutive"):
        merge_utterance([token("評価", 1), token("値", 3)], "評価値")


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_csv_export_preserves_inputs_and_does_not_cross_utterances(tmp_path):
    tokens = tmp_path / "tokens.csv"
    utterances = tmp_path / "utterances.csv"
    output = tmp_path / "tokens_merged.csv"
    _write_csv(tokens, TOKEN_FIELDS, [token("振り", 1, "2023-0001"), token("飛車", 1, "2023-0002")])
    _write_csv(utterances, ["utterance_id", "text"], [
        {"utterance_id": "2023-0001", "text": "振り"},
        {"utterance_id": "2023-0002", "text": "飛車"},
    ])
    before = tokens.read_bytes()
    counts = export_merged_tokens(tokens, utterances, output)
    assert counts == {"utterances": 2, "original_tokens": 2, "merged_tokens": 2,
                      "empty_surface_tokens": 0, "振り飛車": 0, "評価値": 0}
    assert tokens.read_bytes() == before
    with output.open(encoding="utf-8-sig", newline="") as fp:
        result = list(csv.DictReader(fp))
    assert [r["surface"] for r in result] == ["振り", "飛車"]


def test_export_rejects_overwrite_and_unknown_uid(tmp_path):
    tokens = tmp_path / "tokens.csv"
    utterances = tmp_path / "utterances.csv"
    _write_csv(tokens, TOKEN_FIELDS, [token("評価", 1)])
    _write_csv(utterances, ["utterance_id", "text"], [
        {"utterance_id": "2023-9999", "text": "評価"},
    ])
    with pytest.raises(ValueError, match="different"):
        export_merged_tokens(tokens, utterances, tokens)
    with pytest.raises(ValueError, match="Unknown"):
        export_merged_tokens(tokens, utterances, tmp_path / "merged.csv")


def test_real_sudachi_merging_when_installed():
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_core")
    from shogi_analysis.tokenizer import load_sudachi, tokenize_utterance

    analyzer, modes = load_sudachi()
    text = "振り飛車と評価値と角換わり"
    utterance = {
        "year": "2023", "utterance_id": "2023-0001", "role": "player",
        "speaker": "杉本", "text": text,
    }
    original = list(tokenize_utterance(utterance, analyzer, modes.C))
    result, counts = merge_utterance(original, text)
    assert counts["振り飛車"] == 1
    assert counts["評価値"] == 1
    assert [row["surface"] for row in result if row["surface"] in
            {"振り飛車", "評価値", "角換わり"}] == ["振り飛車", "評価値", "角換わり"]

def test_empty_surface_token_is_preserved_but_cannot_bridge_merge():
    # Input is aligned literally, yet a zero-length Sudachi token appears
    # between the two halves of the word. It must not be silently dropped.
    before = [token("振り", 1), token("", 2), token("飛車", 3),
              token("と", 4), token("評価", 5), token("値", 6)]
    after, counts = merge_utterance(before, "振り飛車と評価値")
    assert [r["surface"] for r in after] == ["振り", "", "飛車", "と", "評価値"]
    assert [r["source_token_start"] for r in after] == ["1", "2", "3", "4", "5"]
    assert after[1]["was_merged"] == 0
    assert counts == {"評価値": 1}


def test_empty_surface_token_survives_csv_export(tmp_path):
    tokens = tmp_path / "tokens.csv"
    utterances = tmp_path / "utterances.csv"
    output = tmp_path / "tokens_merged.csv"
    _write_csv(tokens, TOKEN_FIELDS, [token("振り", 1), token("", 2), token("飛車", 3)])
    _write_csv(utterances, ["utterance_id", "text"], [
        {"utterance_id": "2023-0001", "text": "振り飛車"},
    ])
    result = export_merged_tokens(tokens, utterances, output)
    assert result["empty_surface_tokens"] == 1
    assert result["merged_tokens"] == 3
    assert result["振り飛車"] == 0
    with output.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["surface"] for r in rows] == ["振り", "", "飛車"]
