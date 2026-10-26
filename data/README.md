# DDL SLM Data Pipeline & Dataset Architecture

This directory contains the complete data hierarchy for the scratch-trained Natural Language to Database Structure (DDL) Small Language Model (SLM).

## Directory Layout

```
data/
├── raw/                                  # Pristine external downloads & official documentation
│   ├── database_docs/                    # Official dialect reference manuals (normalized text/markdown)
│   │   ├── postgresql/                   # 24 core DDL pages (409 KB, ~102k tokens)
│   │   ├── mysql/                        # 8 core DDL pages (2.44 MB, ~624k tokens)
│   │   ├── sqlite/                       # 16 core DDL pages (194 KB, ~48k tokens)
│   │   └── dbml/                         # 4 syntax & spec pages (31 KB, ~7k tokens)
│   ├── sqale_full/                       # 33 Parquet shards, 511,630 records, 135k schemas (3.37 GB)
│   ├── sql_create_context/               # 78,577 records with CREATE TABLE contexts (16.6 MB)
│   ├── spider/                           # 8,034 cross-domain multi-table records (5.2 MB)
│   ├── clinc_oos/                        # 23,850 intent & out-of-scope records (4.1 MB)
│   └── wikitext103_raw/                  # 1.8M general English rows (~140M tokens, 300.9 MB)
│
├── derived/                              # Normalized & intermediate extractions
│   ├── extracted_schemas/                # Structured database schemas parsed from SQaLe & Spider
│   ├── normalized_sql/                   # Standardized SELECT queries used for context
│   ├── normalized_ddl/                   # Canonical SQLGlot-verified DDL representations
│   ├── dialect_pairs/                    # Cross-dialect conversions (PG <-> MySQL <-> SQLite <-> DBML)
│   └── generated_paraphrases/            # NL request variations across 20 domains
│
└── processed/                            # Final tokenized & partitioned dataset artifacts
    ├── pretrain_train.jsonl              # Database-domain pretraining stream
    ├── pretrain_val.jsonl                # Validation pretraining tokens
    ├── ddl_instruction_train.jsonl       # Multi-turn DDL instruction fine-tuning
    ├── ddl_instruction_val.jsonl         # Held-out schema validation split
    ├── ddl_instruction_test.jsonl        # Evaluation split
    ├── intent_train.jsonl                # 5-class intent classifier training
    ├── safety_train.jsonl                # Destructive / restricted request defense
    └── blind_evaluation.jsonl            # 15k completely unseen domain evaluations
```

## Raw Sources Audit

| # | Source | Path | Size | Records / Tokens | Primary Utility |
|---|--------|------|------|------------------|-----------------|
| 1 | **Official Database Docs** | `data/raw/database_docs/` | 3.07 MB | 52 files, ~782k tokens | DDL syntax, types, constraints, defaults, indexes, dialect nuances |
| 2 | **SQaLe (Full)** | `data/raw/sqale_full/` | 3.37 GB | 511,630 records (33 shards) | 135k real-world multi-table schemas, foreign keys |
| 3 | **SQL Create Context** | `data/raw/sql_create_context/` | 16.6 MB | 78,577 records | Schema understanding, column linking, table terminology |
| 4 | **Spider** | `data/raw/spider/` | 5.2 MB | 8,034 records (200 DBs) | Multi-table joins, cross-domain schemas, unseen test DBs |
| 5 | **CLINC OOS** | `data/raw/clinc_oos/` | 4.1 MB | 23,850 records | General out-of-scope / intent classification baseline |
| 6 | **WikiText-103** | `data/raw/wikitext103_raw/` | 300.9 MB | 1,809,468 rows (~140M tokens) | Auxiliary English pretraining (balanced against DB tokens) |

## Target Record Specification (220,000 DDL Records)

Every record generated in `data/processed/` adheres to:

```json
{
  "id": "ddl_hr_employee_000001",
  "source": "project_authored",
  "license": "original",
  "domain": "human_resources",
  "dialect": "postgresql",
  "task": "generate_ddl",
  "request": "Create an employee table with employee_id, full_name, email, and department.",
  "schema_context": "",
  "expected_status": "accepted",
  "intent": "IN_SCOPE",
  "expected_ddl": "CREATE TABLE employees (...)",
  "validation_status": "valid",
  "schema_group": "hr_employee_v1"
}
```
