# Phase 2 / Step 2：発言単位のTF-IDFと年別単語頻度

## 目的と分析範囲

`utterances.csv` と `tokens_merged.csv` から、棋士の発言（`role=player`）だけを使って、年別の高スコア語と単純な単語頻度を比較する。質問者の発言は、IDFの計算にも含めない。

**現段階ではAI関連発言の意味分類、共起ネットワーク、グラフ作成は行わない。** 高いTF-IDFスコアが、AIに肯定的／否定的な評価を意味するわけではない。

## 実行

Python 3.10以降、リポジトリ直下から以下を実行する（Step 1.5までのCSVが生成済みであること）。

```bash
python -m pip install -e ".[nlp,analysis,dev]"
python -m shogi_analysis.tfidf
python -m pytest -v tests/test_tfidf.py
```

独立した分析用の追加依存は `scikit-learn>=1.4,<2`。未導入の場合は上のpipコマンドで追加する。

主な引数：

```bash
python -m shogi_analysis.tfidf \
  --utterances outputs/tables/utterances.csv \
  --tokens outputs/tables/tokens_merged.csv \
  --output-dir outputs/tables \
  --top-n 20
```

## 前処理

- 棋士の発言を**1発言＝1文書**とする。発言ID、年、棋士名、出典ファイル、誌面ページは保持する。
- `is_content_word=1`かつ`pos1`が「名詞・動詞・形容詞」の語だけ採用する。
- `normalized_form`をUnicode NFKCで正規化して分析語にする。原文の表記や元CSVは変更しない。
- 空欄・数字のみの語・句読点や記号だけの語を除外する。日本語の**1文字語は一律除外しない**。
- ストップワードは**初期状態では設定しない**。後で語彙を確認して除外根拠を明示するため。
- 有効語が0語になった発言も、年別TF-IDF平均の分母（発言数）に含める。

## TF-IDF仕様

`scikit-learn`の`TfidfVectorizer`に**既に形態素解析済みの単語列**を渡す（callable analyzer）。全年度の棋士発言を1つの文書集合としてIDFを一度だけ推定する。

- `smooth_idf=True`：`idf(t)=log((1+N)/(1+df(t)))+1`。
- `norm='l2'`：各発言のTF-IDFベクトルをL2正規化。
- `sublinear_tf=False`、`min_df=1`、`max_df=1.0`：標準設定。
- 年別TF-IDFスコア：その年の**全棋士発言**における各単語のTF-IDFの平均（出現しない発言は0）。これを降順で並べる。

TF-IDFの平均値は、その年にどの語が強く現れたかの探索的指標。**年固有語を因果的に特定する統計検定ではない**。また、「特徴的」の語はこの定義に限定する。

## 出力（UTF-8 BOM付きCSV）

| ファイル | 内容 |
| --- | --- |
| `outputs/tables/tfidf_utterances.csv` | 棋士の各発言×出現語。`utterance_id`、年、棋士名、出典、ページ、発言内順位、TF-IDFスコア、発言内出現回数。語がない発言は行がない。原文そのものは再公開しない。 |
| `outputs/tables/tfidf_year_top_terms.csv` | 年別TF-IDF上位20語（`--top-n`で変更可）。平均TF-IDFと、同じ年の頻度順位・出現回数・1万語あたり頻度などを併記。 |
| `outputs/tables/term_frequency.csv` | 年別の全分析語の単純頻度ランキング。出現回数、語が出た発言数、有効語1万語あたり出現回数を併記。 |

頻度の分母は**その年の棋士発言に含まれる、フィルタ後の有効形態素の総数**である。Phase 1の「1万字あたり頻度」と混同しないこと。

## ストップワード感度分析

結果を見た後に候補語を1行につき1語のテキストファイルに記載し、**別の出力ディレクトリ**に再計算する。

```bash
# 例: stopwords_candidate.txt に「為る」などを1行ずつ記載
python -m shogi_analysis.tfidf \
  --stopwords stopwords_candidate.txt \
  --output-dir outputs/tables/tfidf_stopwords
```

その際、分析語が減るためIDFも変わる。語の除外前後のランキングを比較し、除外語のリストも研究ノートに保存する。恣意的な語選択を避けるため、最初はストップワードなしの出力を基準とする。

## 限界・検証

- 4年間で各年1記事、話し手や質問内容が異なる。スコアの差を「将棋界全体の認識変化」と断定しない。
- 特徴語の意味は、`utterance_id`を使って原文・質問文脈へ戻り、手作業で検証する必要がある。
- 形態素解析とOCRの精度・Phase 1の発言抽出精度に依存する。
- `pytest`は人工データで、質問者除外、年別集計、共通IDFとL2正規化、ストップワード、原本CSV不変更、入力検査を確認する。実際の4年分のランキングはユーザーのローカル実行で確認する。
- 元CSVは変更せず、出力の3ファイルだけを作る。`--output-dir`に既存の同名結果がある場合は上書きするため、条件比較では別ディレクトリを使う。
