"""Unit tests use a stub to test CSV processing without an external dictionary."""
import csv

import pytest

from shogi_analysis.tokenizer import export_tokens, load_sudachi, tokenize_utterance


class FakeMorpheme:
    def __init__(self, surface, base, normalized, pos):
        self._surface = surface
        self._base = base
        self._normalized = normalized
        self._pos = (pos, "一般", "*", "*", "*", "*")

    def surface(self):
        return self._surface

    def dictionary_form(self):
        return self._base

    def normalized_form(self):
        return self._normalized

    def part_of_speech(self):
        return self._pos


class FakeAnalyzer:
    def tokenize(self, text, mode):
        assert mode == "C"
        assert text == "ＡＩを研究した。"
        return [
            FakeMorpheme("ＡＩ", "ＡＩ", "ＡＩ", "名詞"),
            FakeMorpheme("を", "を", "を", "助詞"),
            FakeMorpheme("研究", "研究", "研究", "名詞"),
            FakeMorpheme("し", "する", "する", "動詞"),
            FakeMorpheme("た", "た", "た", "助動詞"),
            FakeMorpheme("。", "。", "。", "補助記号"),
        ]


def test_tokenize_utterance_keeps_every_token_and_normalizes_only_copy():
    utterance = {"year": "2023", "utterance_id": "2023-0001", "role": "player",
                 "speaker": "杉本", "text": "ＡＩを研究した。"}
    tokens = list(tokenize_utterance(utterance, FakeAnalyzer(), "C"))
    assert len(tokens) == 6
    assert [t["token_index"] for t in tokens] == list(range(1, 7))
    assert tokens[0]["surface"] == "ＡＩ"  # source spelling preserved
    assert tokens[0]["normalized_form"] == "AI"
    assert tokens[1]["is_content_word"] == 0
    assert tokens[2]["is_content_word"] == 1
    assert tokens[3]["dictionary_form"] == "する"
    assert tokens[5]["is_content_word"] == 0


def test_export_tokens_to_csv(tmp_path):
    source = tmp_path / "utterances.csv"
    out = tmp_path / "tokens.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["year", "utterance_id", "role", "speaker", "text",
                                                     "source_file", "printed_page_start"])
        writer.writeheader()
        writer.writerow({"year": "2023", "utterance_id": "2023-0001", "role": "player",
                         "speaker": "杉本", "text": "ＡＩを研究した。", "source_file": "2023.txt",
                         "printed_page_start": "39"})
    count = export_tokens(source, out, FakeAnalyzer(), "C")
    assert count == {"utterances": 1, "tokens": 6}
    with out.open(encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert rows[0]["source_file"] == "2023.txt"
    assert rows[0]["printed_page_start"] == "39"
    assert rows[0]["normalized_form"] == "AI"
    assert rows[-1]["surface"] == "。"


def test_export_rejects_missing_columns_and_same_path(tmp_path):
    source = tmp_path / "in.csv"
    source.write_text("year,text\n2023,研究\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different"):
        export_tokens(source, source, FakeAnalyzer(), "C")
    with pytest.raises(ValueError, match="Missing required columns"):
        export_tokens(source, tmp_path / "out.csv", FakeAnalyzer(), "C")


def test_real_sudachi_when_installed():
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_core")
    analyzer, split_modes = load_sudachi()
    result = list(tokenize_utterance(
        {"year": "2023", "utterance_id": "2023-0001", "role": "player",
         "speaker": "杉本", "text": "ＡＩを研究する棋士"}, analyzer, split_modes.C))
    assert any(t["normalized_form"] == "研究" for t in result)
    assert all(t["utterance_id"] == "2023-0001" for t in result)
