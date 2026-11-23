# Phase 2 / Step 1.5：将棋用語の複合語統合

## 目的
SudachiPy（SplitMode C）が実際に「振り飛車」→「振り」「飛車」、「評価値」→「評価」「値」と分割することを確認したため、**研究上の集計語彙を揃える**ための限定的な後処理を追加する。

今回の変換規則は、上記2語だけ（`src/shogi_analysis/term_merger.py` の `MERGE_RULES`）。それ以外の辞書形・品詞・原文は変更しない。

## 実行方法（Python 3.10以上）
Phase 1とPhase 2 Step 1で `utterances.csv` と `tokens.csv` を作成済みの状態で：

```bash
python -m shogi_analysis.term_merger \
  --tokens outputs/tables/tokens.csv \
  --utterances outputs/tables/utterances.csv \
  --output outputs/tables/tokens_merged.csv
python -m pytest -v tests/test_term_merger.py
```

`tokens.csv`は読み込み専用。新しく `tokens_merged.csv` を生成するため、処理前後を比較できる。

## 統合条件
- **同じ発言内**の隣接トークンに限る。発言をまたぐことはない。
- 単語の `surface` が厳密に `振り`＋`飛車` または `評価`＋`値` と一致すること。
- Phase 1の `utterances.csv` にある**原文で、2つの語が間に空白・句読点等を挟まず連続**していること。
- 単語列が原文と照合できない場合はエラーを返し、無理に統合しない。
- `角換わり` など元から1単語になっている語は変更しない。

## CSVの列
既存の`tokens.csv`列に、次の4列を追加する。

| 列 | 意味 |
| --- | --- |
| `source_token_start` | 元の `tokens.csv` の開始 `token_index` |
| `source_token_end` | 元の `tokens.csv` の終了 `token_index` |
| `was_merged` | 結合後の語なら1、無加工なら0 |
| `merge_rule` | 適用した統合語（例：`評価値`）、無加工なら空欄 |

結合後の `token_index` は**結合後の列の順序に振り直す**。元のインデックスは`source_token_start` / `source_token_end` に保持する。統合後の語の品詞は「名詞／普通名詞／一般」、内容語フラグは1とする。

## テストと限界
- 単体テストは、語の統合・空白がある場合の非統合・発言境界・原文との照合・入力ファイル保全を確認。
- SudachiPyの実辞書による統合確認テストは、辞書がインストールされている環境で実行。辞書がない環境では`skip`になる。
- 実際の4年分の統合件数はローカルで上記コマンドを実行して確認すること（ここでは実辞書を使った全集計は未検証）。
- 複合語を人為的に設定するため、TF-IDFでは**統合前後の結果も比較**し、語彙の選定によるバイアスを検討する。

今回の範囲にTF-IDFや共起分析は含めない。

## 空の surface に対する例外処理（追加）

SudachiPy由来の形態素には `surface == ""` の行が含まれる場合がある。データ欠損と断定して削除せず、
原文との位置合わせでは長さ0の区間に対応させ、その行を出力にも保持する。

- 空文字形態素の前後を飛び越えて、別々のトークンを結合することはない。
- 実行結果の `empty_surface_tokens` に、処理対象に含まれた空文字トークン数を表示する。
- 原文に対して非空トークンを正しく位置合わせできない場合は従来どおりエラーにする。
- CSVが最後まで処理できたことは、原文のOCR精度やSudachiの形態素分割精度を保証しない。
