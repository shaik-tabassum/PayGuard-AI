"""
backend/main.py
----------------
PayGuard AI API.

Run from the project root with:
    uvicorn backend.main:app --reload

Docs (Swagger UI):
    http://127.0.0.1:8000/docs

FIXES FROM THE ORIGINAL VERSION
================================
- No hardcoded Windows-specific paths - everything is resolved relative to
  this file with pathlib, so it runs the same on Windows/macOS/Linux and on
  any teammate's machine or a deployment server.
- The risk engine and its trained model are loaded ONCE at startup (not
  once per request), which is both faster and fails fast with a clear
  error if the model hasn't been trained yet, instead of crashing on the
  first request.
- Pydantic validation now rejects negative amounts / obviously invalid
  fields at the API boundary instead of silently passing bad data to the
  model.
- Every route is wrapped so a bad request returns a proper 4xx/5xx JSON
  error instead of an unhandled 500 with a stack trace leaking to the
  client.
- Added SQLite-backed transaction history (Phase 6) and analytics summary
  (Phase 7) endpoints that the dashboard consumes.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make the sibling `ml/` package importable regardless of the working
# directory the app is launched from.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "ml"))

from risk_engine import ModelNotLoadedError, get_engine  # noqa: E402

DB_PATH = ROOT_DIR / "data" / "payguard_history.db"

app = FastAPI(
    title="PayGuard AI",
    description="AI-powered transaction risk detection and automated risk assessment API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype only - restrict to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    payment_method: Literal["Card", "UPI", "NetBanking", "Wallet"]
    location: str = Field(..., min_length=1, max_length=100)
    device_type: Literal["Mobile", "Desktop", "Tablet"]
    new_device: Literal[0, 1]
    location_change: Literal[0, 1]
    transactions_last_hour: int = Field(..., ge=0, le=1000)
    account_age_days: int = Field(..., ge=0, le=50000)
    previous_fraud_count: int = Field(..., ge=0, le=1000)
    customer_id: str | None = Field(default=None, max_length=50)
    transaction_id: str | None = Field(default=None, max_length=50)

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 100000,
                "payment_method": "Card",
                "location": "Delhi",
                "device_type": "Mobile",
                "new_device": 1,
                "location_change": 1,
                "transactions_last_hour": 10,
                "account_age_days": 20,
                "previous_fraud_count": 2,
            }
        }
    }


class RiskResponse(BaseModel):
    ml_score: float
    rule_score: float
    risk_score: float
    risk_level: str
    decision: str
    reasons: list[str]
    warnings: list[str]
    timestamp: str


# --------------------------------------------------------------------------
# SQLite history store
# --------------------------------------------------------------------------
@contextmanager
def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                transaction_id TEXT,
                customer_id TEXT,
                amount REAL NOT NULL,
                payment_method TEXT NOT NULL,
                location TEXT NOT NULL,
                device_type TEXT NOT NULL,
                ml_score REAL NOT NULL,
                rule_score REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                decision TEXT NOT NULL,
                reasons TEXT NOT NULL
            )
            """
        )


def save_transaction(txn: Transaction, result: dict, timestamp: str) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO transactions
                (timestamp, transaction_id, customer_id, amount, payment_method,
                 location, device_type, ml_score, rule_score, risk_score,
                 risk_level, decision, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                txn.transaction_id,
                txn.customer_id,
                txn.amount,
                txn.payment_method,
                txn.location,
                txn.device_type,
                result["ml_score"],
                result["rule_score"],
                result["risk_score"],
                result["risk_level"],
                result["decision"],
                " | ".join(result["reasons"]),
            ),
        )


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    init_db()
    try:
        get_engine()
    except ModelNotLoadedError as exc:
        # Don't crash the whole app at import time - surface a clear error
        # on first use instead, so `uvicorn backend.main:app` at least
        # boots and /docs still explains what's wrong.
        print(f"[PayGuard AI] WARNING: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "PayGuard AI",
        "status": "online",
        "docs": "/docs",
        "endpoints": ["/analyze", "/history", "/analytics"],
    }


@app.post("/analyze", response_model=RiskResponse)
def analyze(txn: Transaction):
    try:
        engine = get_engine()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        result = engine.assess(txn.model_dump())
    except Exception as exc:  # noqa: BLE001 - convert any scoring failure into a clean 500
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {exc}")

    timestamp = datetime.now(timezone.utc).isoformat()
    payload = result.to_dict()

    try:
        save_transaction(txn, payload, timestamp)
    except Exception as exc:  # noqa: BLE001 - history logging must never break the response
        print(f"[PayGuard AI] Failed to save transaction history: {exc}", file=sys.stderr)

    return {**payload, "timestamp": timestamp}


@app.get("/history")
def history(limit: int = 20):
    limit = max(1, min(limit, 200))
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"count": len(rows), "transactions": [dict(row) for row in rows]}


@app.get("/analytics")
def analytics():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM transactions").fetchone()["c"]
        by_level = conn.execute(
            "SELECT risk_level, COUNT(*) AS c FROM transactions GROUP BY risk_level"
        ).fetchall()
        by_decision = conn.execute(
            "SELECT decision, COUNT(*) AS c FROM transactions GROUP BY decision"
        ).fetchall()
        avg_score = conn.execute(
            "SELECT AVG(risk_score) AS a FROM transactions"
        ).fetchone()["a"]

    level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for row in by_level:
        level_counts[row["risk_level"]] = row["c"]

    decision_counts = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}
    for row in by_decision:
        decision_counts[row["decision"]] = row["c"]

    return {
        "total_transactions": total,
        "low_risk": level_counts["LOW"],
        "medium_risk": level_counts["MEDIUM"],
        "high_risk": level_counts["HIGH"],
        "blocked": decision_counts["BLOCK"],
        "average_risk_score": round(avg_score, 2) if avg_score is not None else 0,
    }
