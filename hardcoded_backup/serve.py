"""Standalone API & Web UI Server for the Hardcoded DDL Prototype.

Allows demonstrating the hardcoded/rule-based results via both:
  1. A modern Web UI (accessible at http://127.0.0.1:8001)
  2. REST API endpoints (compatible with Swagger docs at http://127.0.0.1:8001/docs)

Usage:
  python hardcoded_backup/serve.py
  python hardcoded_backup/serve.py --port 8001 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Ensure project root and hardcoded_backup are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from hardcoded_backup.demo import run_hardcoded_pipeline

app = FastAPI(
    title="SQL DDL Prototype (Hardcoded / Rule-Based)",
    description="Dedicated server showcasing the deterministic, hardcoded prototype generator.",
    version="0.1.0-prototype",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    requirement: str
    dialect: str | None = None
    task: str = "generate_ddl"


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hardcoded DDL Prototype Demo</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', -apple-system, sans-serif;
      background: radial-gradient(circle at top left, #1e1b4b 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      padding: 32px 20px;
    }
    .container { max-width: 960px; margin: 0 auto; }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-radius: 9999px;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.4);
      margin-bottom: 12px;
    }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; }
    p.subtitle { color: var(--text-muted); margin-bottom: 28px; font-size: 1.05rem; }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .form-group { margin-bottom: 16px; }
    label { display: block; font-weight: 600; margin-bottom: 8px; font-size: 0.9rem; color: #cbd5e1; }
    textarea, select {
      width: 100%;
      padding: 12px 16px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 10px;
      color: #fff;
      font-family: inherit;
      font-size: 1rem;
      outline: none;
      transition: all 0.2s;
    }
    textarea:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .preset-btn {
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      color: #cbd5e1;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .preset-btn:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent);
      color: #fff;
    }
    .btn-submit {
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      border: none;
      padding: 12px 24px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 1rem;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: transform 0.1s, background 0.2s;
    }
    .btn-submit:hover { background: #4f46e5; transform: translateY(-1px); }
    .btn-submit:active { transform: translateY(1px); }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .meta-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
    }
    .meta-title { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .meta-val { font-size: 1.1rem; font-weight: 700; margin-top: 4px; }
    pre {
      background: #050811;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 16px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      color: #38bdf8;
      overflow-x: auto;
      white-space: pre-wrap;
    }
    .status-accepted { color: var(--success); }
    .status-rejected { color: var(--danger); }
    .status-warn { color: var(--warning); }
  </style>
</head>
<body>
  <div class="container">
    <span class="badge">Archive / Prototype Demonstration</span>
    <h1>Deterministic SQL DDL Prototype</h1>
    <p class="subtitle">Execute and audit the hardcoded schema compiler and heuristic scope gate.</p>

    <div class="card">
      <div class="form-group">
        <label>Quick Presets (Click to test):</label>
        <div class="presets">
          <button class="preset-btn" onclick="setPreset('Create an employee table with id, name, and salary in PostgreSQL', 'postgresql')">Employee (PG)</button>
          <button class="preset-btn" onclick="setPreset('Create a customers table with id bigint, email varchar(255) unique, and created_at in MySQL', 'mysql')">Customers (MySQL)</button>
          <button class="preset-btn" onclick="setPreset('Create a products table with sku text primary key, title text, and price real in SQLite', 'sqlite')">Products (SQLite)</button>
          <button class="preset-btn" onclick="setPreset('Create a users table with id int, username varchar(50), and role varchar(20) in DBML', 'dbml')">Users (DBML)</button>
          <button class="preset-btn" onclick="setPreset('Add phone varchar(32) to table customers', 'postgresql')">Alter Table</button>
          <button class="preset-btn" onclick="setPreset('Create an index on customers (email)', 'postgresql')">Create Index</button>
          <button class="preset-btn" onclick="setPreset('Drop all tables in production', 'postgresql')">Restricted Drop</button>
          <button class="preset-btn" onclick="setPreset('What is machine learning?', 'postgresql')">Out of Scope</button>
        </div>
      </div>

      <div class="form-group">
        <label for="req">DDL Requirement:</label>
        <textarea id="req" rows="3" placeholder="e.g. Create a table employee with id, name, and salary in PostgreSQL"></textarea>
      </div>

      <div class="form-group" style="display: flex; gap: 16px; align-items: flex-end;">
        <div style="flex: 1;">
          <label for="dialect">Dialect:</label>
          <select id="dialect">
            <option value="postgresql">PostgreSQL (Default)</option>
            <option value="mysql">MySQL</option>
            <option value="sqlite">SQLite</option>
            <option value="dbml">DBML</option>
          </select>
        </div>
        <button class="btn-submit" onclick="submitQuery()">Compile DDL &#9658;</button>
      </div>
    </div>

    <div id="result-card" class="card" style="display: none;">
      <h3 style="margin-bottom: 16px;">Generation Output & Diagnostics</h3>
      <div class="meta-grid">
        <div class="meta-box">
          <div class="meta-title">Status</div>
          <div id="res-status" class="meta-val">ACCEPTED</div>
        </div>
        <div class="meta-box">
          <div class="meta-title">Intent / Scope</div>
          <div id="res-intent" class="meta-val">IN_SCOPE</div>
        </div>
        <div class="meta-box">
          <div class="meta-title">Compiler Engine</div>
          <div id="res-engine" class="meta-val">local_schema_compiler</div>
        </div>
        <div class="meta-box">
          <div class="meta-title">Latency</div>
          <div id="res-latency" class="meta-val">0.0 ms</div>
        </div>
      </div>

      <div style="margin-bottom: 16px;">
        <label>Generated DDL Statement:</label>
        <pre id="res-sql">--</pre>
      </div>

      <div>
        <label>Extracted Schema AST Plan (JSON):</label>
        <pre id="res-plan" style="color: #a78bfa; max-height: 250px;">--</pre>
      </div>
    </div>
  </div>

  <script>
    function setPreset(text, dialect) {
      document.getElementById('req').value = text;
      document.getElementById('dialect').value = dialect;
      submitQuery();
    }

    async function submitQuery() {
      const requirement = document.getElementById('req').value.trim();
      const dialect = document.getElementById('dialect').value;
      if (!requirement) return;

      const card = document.getElementById('result-card');
      card.style.display = 'block';

      try {
        const resp = await fetch('/api/v1/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ requirement, dialect })
        });
        const data = await resp.json();

        const statusEl = document.getElementById('res-status');
        statusEl.innerText = data.status.toUpperCase();
        statusEl.className = 'meta-val ' + (data.status === 'accepted' ? 'status-accepted' : 'status-rejected');

        document.getElementById('res-intent').innerText = data.intent;
        document.getElementById('res-engine').innerText = data.source || 'scope_gate';
        document.getElementById('res-latency').innerText = data.latency_ms + ' ms';

        document.getElementById('res-sql').innerText = data.sql || (data.message || 'No SQL generated.');
        document.getElementById('res-plan').innerText = data.plan ? JSON.stringify(data.plan, null, 2) : 'No plan generated.';
      } catch (err) {
        alert('Request failed: ' + err.message);
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "hardcoded_prototype"}


@app.post("/api/v1/generate")
def generate(payload: GenerateRequest) -> dict:
    if not payload.requirement.strip():
        raise HTTPException(status_code=400, detail="Requirement cannot be empty")
    return run_hardcoded_pipeline(payload.requirement, payload.dialect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hardcoded prototype API and Web UI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    args = parser.parse_args()

    print(f"\n" + "=" * 65)
    print(f"  Starting Hardcoded DDL Prototype Server")
    print(f"  Web UI   : http://{args.host}:{args.port}/")
    print(f"  Swagger  : http://{args.host}:{args.port}/docs")
    print(f"  API POST : http://{args.host}:{args.port}/api/v1/generate")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
