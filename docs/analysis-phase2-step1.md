# Phase 2 / Step 1: SudachiPyによる形態素解析

Phase 1で生成した `utterances.csv` を単語（形態素）単位に展開する。**この段階ではTF-IDF・共起分析は実施しない**。

## セットアップ

Python 3.10以降でリポジトリのルートから：

```bash
python -m pip install -e ".[nlp,dev]"
python -m shogi_analysis --data-dir data/raw --output-dir outputs/tables
python -m shogi_analysis.tokenizer --input outputs/tables/utterances.csv --output outputs/tables/tokens.csv --mode C
python -m pytest -v
```

- `sudachipy==0.6.10`、`sudachidict_core==20260428`を指定する。SudachiPy 0.7系と新辞書のバイナリ形式変更による互換性問題を避けるため。
- 形態素解析は `SplitMode.C`（複合語を比較的まとめる粒度）を初期値とする。`--mode A`や`--mode B`に変えて比較もできる。
- コードは `src/shogi_analysis/tokenizer.py`、実行入口は `python -m shogi_analysis.tokenizer`。

## 出力：`outputs/tables/tokens.csv`

1行＝1形態素。各行は `year`、`utterance_id`、`role`、`speaker`、`source_file`、`printed_page_start`、`token_index` により元発言へ追跡できる。

- `surface`: 原文の表記（例：`ＡＩ`）。
- `dictionary_form`: 辞書形（動詞の活用形を基本形にできる）。
- `normalized_form`: Sudachi辞書の正規化形をさらにUnicode NFKC正規化した語（例：`AI`）。**原文を書き換えない**。
- `pos1`～`pos4`: Sudachiの品詞情報。
- `is_content_word`: 最上位の品詞が名詞・動詞・形容詞なら1、それ以外は0。

**助詞・助動詞・記号も削除せず出力する**。`is_content_word`は将来のTF-IDFなどに使う候補フラグであり、研究上の除外条件は今後確認して決める。

## 検証方針と限界

- pytestのスタブを用いた単体テストで表記保持、正規化、品詞フラグ、出力形式、入力検証を確認する。
- SudachiPyとCore辞書がインストールされている環境では、実辞書の統合テストも実行する。未インストールならこのテストはskipされる。
- 語の分割の正しさは辞書の精度や分割モードに依存する。将棋固有語（「角換わり」「振り飛車」等）の分割結果を実際に確認し、必要ならカスタム辞書や後処理を検討する。
- Phase 1の発言抽出ミスや未校正箇所はこの工程では修正されない。`data/raw`とPhase 1の実装は変更しない。
