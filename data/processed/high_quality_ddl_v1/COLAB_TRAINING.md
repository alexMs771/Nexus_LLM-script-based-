# High-quality DDL/DBML Colab package

This package is the active compact dataset. It is intentionally source-aware and
not randomly balanced. Raw data remains outside this package and is not needed
by the first supervised training run.

## First run: encoder-decoder SQL DDL model used by the API

This is the recommended architecture. The encoder reads the full request and
the decoder generates the DDL answer token by token with cross-attention.

Run from the project root after copying/unzipping this dataset into the Colab
project:

```bash
python -m training.train_seq2seq \
  --config training/configs/seq2seq.json \
  --train-path data/processed/high_quality_ddl_v1/supervised/ddl_sql/train.jsonl \
  --val-path data/processed/high_quality_ddl_v1/supervised/ddl_sql/val.jsonl \
  --out-dir artifacts/models/own-ddl-encoder-decoder-v1 \
  --device cuda \
  --max-steps 8000
```

The API automatically prefers `artifacts/models/own-ddl-encoder-decoder-v1`
when that trained artifact exists; otherwise it keeps the legacy model path.

Evaluate only on the held-out test set:

```bash
python -m training.evaluate_seq2seq \
  --model-dir artifacts/models/own-ddl-encoder-decoder-v1 \
  --test-path data/processed/high_quality_ddl_v1/supervised/ddl_sql/test.jsonl \
  --candidates 3 \
  --out artifacts/models/own-ddl-encoder-decoder-v1/evaluation_seq2seq.json
```

The API must use the resulting artifact only after the evaluation report is
reviewed. A successful training run is not the same as acceptable SQL quality;
the required checks are syntax validity, exact/normalized match, operation
coverage, and disposable-database execution where available.

## Separate DBML model

DBML is a different output language. If DBML is the deployment default, train a
separate artifact rather than mixing SQL and DBML targets in one generator:

```bash
python -m training.train_seq2seq \
  --config training/configs/seq2seq.json \
  --train-path data/processed/high_quality_ddl_v1/supervised/dbml/train.jsonl \
  --val-path data/processed/high_quality_ddl_v1/supervised/dbml/val.jsonl \
  --out-dir artifacts/models/own-dbml-encoder-decoder-v1 \
  --device cuda \
  --max-steps 8000
```

`out_of_scope` is retained as a separate intent/policy corpus. It should not be
silently mixed into the SQL generator targets. The service scope gate decides
whether a request is a DDL request before the generator is called.

`sql_support`, `dbml/documents.jsonl`, and `english` are supporting corpora for
later continued pretraining or tokenizer experiments; they are not required for
the first supervised DDL quality run.
