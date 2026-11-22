from pathlib import Path
from shogi_analysis.pipeline import keyword_mentions, parse_interview, run


def test_variants_and_parenthetical_deduplication():
    mentions = keyword_mentions("ＡＩ（人工知能）とソフト、コンピューター、AI")
    assert [m["term"] for m in mentions] == ["AI", "ソフト", "コンピュータ", "AI"]


def test_utterance_crossing_page_and_unlabeled_reply(tmp_path: Path):
    p = tmp_path / "shogi_yearbook_2026_fulltext.txt"
    p.write_text("""【PDF 1ページ目／誌面10ページ】
【本文】
――ＡＩを使っていますか。
「最初のページから
【図・キャプション】
図の説明
---
【PDF 2ページ目／誌面11ページ】
【本文】
続けて使っています」
――どうでしょうか。
「ＡＩ（人工知能）です」
""", encoding="utf-8")
    units = parse_interview(p, 2026)
    assert len(units) == 4
    assert units[1]["speaker"] == "増田"
    assert units[1]["text"] == "最初のページから続けて使っています"
    assert units[1]["printed_page_start"] == "10"
    assert units[1]["printed_page_end"] == "11"
    assert units[1]["question_id"] == units[0]["utterance_id"]
    assert units[3]["mention_count"] == 1


def test_full_corpus_runs_when_present(tmp_path: Path):
    data = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not data.exists():
        return
    counts = run(data, tmp_path)
    assert set(counts) == {2017, 2020, 2023, 2026}
    assert all(n > 0 for n in counts.values())
    assert (tmp_path / "frequency_by_year.csv").is_file()
