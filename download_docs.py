"""Download official database DDL documentation pages.

Fetches the key DDL reference pages from PostgreSQL, MySQL, SQLite, and
DBML documentation. Uses httpx with curl fallback.
Each page is parsed from HTML to clean, structured text/markdown.

Target folders:
  data/raw/database_docs/postgresql/
  data/raw/database_docs/mysql/
  data/raw/database_docs/sqlite/
  data/raw/database_docs/dbml/
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx


class HTMLToMarkdownParser(HTMLParser):
    """Converts HTML documentation to clean readable text/markdown."""

    def __init__(self):
        super().__init__()
        self.pieces: list[str] = []
        self.skip_stack = 0
        self.skip_tags = {"script", "style", "nav", "footer", "header", "noscript", "svg"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag_lower = tag.lower()
        if tag_lower in self.skip_tags:
            self.skip_stack += 1
            return
        if self.skip_stack > 0:
            return

        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag_lower[1])
            self.pieces.append(f"\n\n{'#' * level} ")
        elif tag_lower in ("p", "div", "section", "article"):
            self.pieces.append("\n\n")
        elif tag_lower == "br":
            self.pieces.append("\n")
        elif tag_lower in ("li", "dt"):
            self.pieces.append("\n- ")
        elif tag_lower == "dd":
            self.pieces.append("\n  ")
        elif tag_lower == "pre":
            self.pieces.append("\n\n```\n")
        elif tag_lower == "code":
            self.pieces.append(" `")
        elif tag_lower in ("td", "th"):
            self.pieces.append(" | ")
        elif tag_lower == "tr":
            self.pieces.append("\n")

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self.skip_tags:
            self.skip_stack = max(0, self.skip_stack - 1)
            return
        if self.skip_stack > 0:
            return

        if tag_lower == "pre":
            self.pieces.append("\n```\n\n")
        elif tag_lower == "code":
            self.pieces.append("` ")
        elif tag_lower in ("p", "div", "section", "article", "tr", "table"):
            self.pieces.append("\n")

    def handle_data(self, data: str):
        if self.skip_stack == 0 and data:
            self.pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self.pieces)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" +\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_url(url: str, timeout_sec: int = 25) -> str | None:
    # 1. Try httpx first (works around Akamai TLS fingerprinting on MySQL/Oracle)
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout_sec, verify=False) as client:
            resp = client.get(url)
            if resp.status_code == 200 and len(resp.text) > 500:
                # Check for "Technical Difficulties" fake block page
                if "Technical Difficulties" not in resp.text[:1000]:
                    return resp.text
    except Exception:
        pass

    # 2. Fallback to curl.exe
    cmd = [
        "curl.exe",
        "-s",
        "-L",
        "--max-time", str(timeout_sec),
        "-A", HEADERS["User-Agent"],
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_sec + 5)
        if proc.returncode == 0 and proc.stdout:
            try:
                text = proc.stdout.decode("utf-8")
            except UnicodeDecodeError:
                text = proc.stdout.decode("latin-1", errors="replace")
            if "Technical Difficulties" not in text[:1000]:
                return text
    except Exception:
        pass

    return None


def download_and_clean_page(url: str, dest_file: Path, label: str) -> bool:
    if dest_file.exists() and dest_file.stat().st_size > 1500:
        # Check if existing file has valid content
        content = dest_file.read_text(encoding="utf-8", errors="replace")
        if "Technical Difficulties" not in content[:500]:
            print(f"  [EXISTS] {label} ({dest_file.stat().st_size:,} bytes)", flush=True)
            return True

    print(f"  [FETCH]  {label} <- {url}", flush=True)
    t0 = time.perf_counter()
    raw = fetch_url(url)
    if not raw:
        print(f"  [FAIL]   {label}: empty response or fetch error", flush=True)
        return False

    if url.endswith(".md") or url.endswith(".txt"):
        cleaned = raw
    else:
        parser = HTMLToMarkdownParser()
        parser.feed(raw)
        cleaned = parser.get_text()

    if len(cleaned) < 300 or "Technical Difficulties" in cleaned[:500]:
        print(f"  [FAIL]   {label}: extracted text too short or blocked ({len(cleaned)} chars)", flush=True)
        return False

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_text(cleaned, encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"  [OK]     {label}: saved {len(cleaned):,} chars ({dest_file.stat().st_size:,} bytes) in {elapsed:.1f}s", flush=True)
    return True


# ---------------------------------------------------------------------------
# Documentation URL dictionaries
# ---------------------------------------------------------------------------

PG_BASE = "https://www.postgresql.org/docs/current"
POSTGRESQL_PAGES = {
    "create_table": f"{PG_BASE}/sql-createtable.html",
    "alter_table": f"{PG_BASE}/sql-altertable.html",
    "drop_table": f"{PG_BASE}/sql-droptable.html",
    "create_index": f"{PG_BASE}/sql-createindex.html",
    "drop_index": f"{PG_BASE}/sql-dropindex.html",
    "create_schema": f"{PG_BASE}/sql-createschema.html",
    "drop_schema": f"{PG_BASE}/sql-dropschema.html",
    "create_view": f"{PG_BASE}/sql-createview.html",
    "drop_view": f"{PG_BASE}/sql-dropview.html",
    "create_sequence": f"{PG_BASE}/sql-createsequence.html",
    "create_type": f"{PG_BASE}/sql-createtype.html",
    "create_domain": f"{PG_BASE}/sql-createdomain.html",
    "create_trigger": f"{PG_BASE}/sql-createtrigger.html",
    "create_function": f"{PG_BASE}/sql-createfunction.html",
    "ddl_constraints": f"{PG_BASE}/ddl-constraints.html",
    "data_types": f"{PG_BASE}/datatype.html",
    "ddl_overview": f"{PG_BASE}/ddl.html",
    "ddl_default": f"{PG_BASE}/ddl-default.html",
    "ddl_generated_columns": f"{PG_BASE}/ddl-generated-columns.html",
    "ddl_inherit": f"{PG_BASE}/ddl-inherit.html",
    "ddl_partitioning": f"{PG_BASE}/ddl-partitioning.html",
    "ddl_schemas": f"{PG_BASE}/ddl-schemas.html",
    "ddl_alter": f"{PG_BASE}/ddl-alter.html",
    "ddl_rowsecurity": f"{PG_BASE}/ddl-rowsecurity.html",
}

MY_BASE = "https://dev.mysql.com/doc/refman/8.4/en"
MYSQL_PAGES = {
    "create_table": f"{MY_BASE}/create-table.html",
    "alter_table": f"{MY_BASE}/alter-table.html",
    "drop_table": f"{MY_BASE}/drop-table.html",
    "create_index": f"{MY_BASE}/create-index.html",
    "drop_index": f"{MY_BASE}/drop-index.html",
    "create_view": f"{MY_BASE}/create-view.html",
    "create_database": f"{MY_BASE}/create-database.html",
    "create_trigger": f"{MY_BASE}/create-trigger.html",
    "create_function": f"{MY_BASE}/create-function.html",
    "data_types": f"{MY_BASE}/data-types.html",
    "numeric_types": f"{MY_BASE}/numeric-type-syntax.html",
    "string_types": f"{MY_BASE}/string-type-syntax.html",
    "date_time_types": f"{MY_BASE}/date-and-time-type-syntax.html",
    "spatial_types": f"{MY_BASE}/spatial-data-types.html",
    "json_data_type": f"{MY_BASE}/json.html",
    "constraints": f"{MY_BASE}/constraints.html",
    "foreign_keys": f"{MY_BASE}/create-table-foreign-keys.html",
    "check_constraints": f"{MY_BASE}/create-table-check-constraints.html",
    "generated_columns": f"{MY_BASE}/create-table-generated-columns.html",
    "secondary_indexes": f"{MY_BASE}/create-table-secondary-indexes.html",
    "auto_increment": f"{MY_BASE}/example-auto-increment.html",
}

SQ_BASE = "https://sqlite.org"
SQLITE_PAGES = {
    "create_table": f"{SQ_BASE}/lang_createtable.html",
    "alter_table": f"{SQ_BASE}/lang_altertable.html",
    "drop_table": f"{SQ_BASE}/lang_droptable.html",
    "create_index": f"{SQ_BASE}/lang_createindex.html",
    "drop_index": f"{SQ_BASE}/lang_dropindex.html",
    "create_view": f"{SQ_BASE}/lang_createview.html",
    "drop_view": f"{SQ_BASE}/lang_dropview.html",
    "create_trigger": f"{SQ_BASE}/lang_createtrigger.html",
    "data_types": f"{SQ_BASE}/datatype3.html",
    "foreign_keys": f"{SQ_BASE}/foreignkeys.html",
    "generated_columns": f"{SQ_BASE}/gencol.html",
    "strict_tables": f"{SQ_BASE}/strict.html",
    "without_rowid": f"{SQ_BASE}/withoutrowid.html",
    "auto_increment": f"{SQ_BASE}/autoinc.html",
    "partial_index": f"{SQ_BASE}/partialindex.html",
    "lang_expr": f"{SQ_BASE}/lang_expr.html",
}

DBML_PAGES = {
    "syntax_docs": "https://dbml.dbdiagram.io/docs/",
    "cli_docs": "https://dbml.dbdiagram.io/cli/",
    "github_readme": "https://raw.githubusercontent.com/holistics/dbml/master/README.md",
    "core_spec": "https://raw.githubusercontent.com/holistics/dbml/master/packages/dbml-core/README.md",
}


def run_download():
    base_dir = Path("data/raw/database_docs")
    categories = [
        ("PostgreSQL", POSTGRESQL_PAGES, base_dir / "postgresql"),
        ("MySQL", MYSQL_PAGES, base_dir / "mysql"),
        ("SQLite", SQLITE_PAGES, base_dir / "sqlite"),
        ("DBML", DBML_PAGES, base_dir / "dbml"),
    ]

    total_ok = 0
    total_fail = 0
    total_bytes = 0

    print("=" * 70, flush=True)
    print("  Official Database Documentation Downloader & Normalizer", flush=True)
    print("=" * 70, flush=True)

    for dialect_name, pages, dest_dir in categories:
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {dialect_name} ({len(pages)} pages) -> {dest_dir} ---", flush=True)
        cat_ok = 0
        cat_fail = 0
        for name, url in pages.items():
            suffix = ".md" if url.endswith(".md") else ".txt"
            target_path = dest_dir / f"{name}{suffix}"
            success = download_and_clean_page(url, target_path, f"{dialect_name}/{name}")
            if success:
                cat_ok += 1
                total_bytes += target_path.stat().st_size
            else:
                cat_fail += 1

        total_ok += cat_ok
        total_fail += cat_fail
        print(f"  {dialect_name} Results: {cat_ok} OK, {cat_fail} Failed", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Documentation Download Summary: {total_ok} OK, {total_fail} Failed", flush=True)
    print(f"Total Disk Space used: {total_bytes / (1024 * 1024):.2f} MB", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_download()
