"""
risk_engine.py
---------------
Combines the trained ML model's fraud probability with a transparent,
human-readable rule-based score to produce a final risk assessment.

DESIGN NOTES
============
- The ML score and rule score are computed completely independently, then
  combined with a documented, fixed formula. Neither one is nudged based
  on the other - that would defeat the point of having two independent
  signals, and the brief explicitly asked not to artificially manipulate
  scores.
- Every rule that fires appends a plain-English reason, so the API
  response (and the dashboard) can show *why* a transaction was flagged,
  not just a number.
- All inputs are validated/sanitized defensively. A malformed or missing
  field should degrade gracefully (documented default + a note in the
  reasons list), never raise an unhandled exception that 500s the API.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "fraud_model.pkl"

# ---- Weights for combining ML + rule scores -------------------------------
# Kept as named constants (not magic numbers buried in a formula) so they're
# easy to justify and easy to tune later.
ML_WEIGHT = 0.70
RULE_WEIGHT = 0.30

# ---- Risk level thresholds -------------------------------------------------
LOW_MAX = 30
MEDIUM_MAX = 70
# 0-30 LOW | 31-70 MEDIUM | 71-100 HIGH

DECISION_BY_LEVEL = {
    "LOW": "APPROVE",
    "MEDIUM": "REVIEW",
    "HIGH": "BLOCK",
}


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass
class RiskResult:
    ml_score: float
    rule_score: float
    risk_score: float
    risk_level: str
    decision: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ml_score": round(self.ml_score, 2),
            "rule_score": round(self.rule_score, 2),
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level,
            "decision": self.decision,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


class RiskEngine:
    """Loads the trained model once and reuses it for every request."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.threshold = 0.5
        self.feature_columns: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise ModelNotLoadedError(
                f"No trained model found at {self.model_path}. "
                "Run `python ml/train_model.py` first."
            )
        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)
        self.pipeline = artifact["pipeline"]
        self.threshold = artifact.get("threshold", 0.5)
        self.feature_columns = artifact["feature_columns"]

    # ---------------------------------------------------------------- ML --
    def _sanitize(self, transaction: dict) -> tuple[dict, list[str]]:
        """Fills safe defaults for missing/invalid fields instead of
        crashing, and records what it had to fix so the caller can see it
        in the response's `warnings` list."""
        warnings: list[str] = []
        defaults = {
            "amount": 0.0,
            "payment_method": "Unknown",
            "location": "Unknown",
            "device_type": "Unknown",
            "new_device": 0,
            "location_change": 0,
            "transactions_last_hour": 0,
            "account_age_days": 365,
            "previous_fraud_count": 0,
        }
        clean = dict(transaction) if transaction else {}
        for key, default in defaults.items():
            value = clean.get(key, None)
            if value is None:
                clean[key] = default
                warnings.append(f"Missing '{key}', used default value.")
                continue
            if key in ("amount",):
                try:
                    clean[key] = max(0.0, float(value))
                except (TypeError, ValueError):
                    clean[key] = default
                    warnings.append(f"Invalid '{key}', used default value.")
            elif key in ("new_device", "location_change", "transactions_last_hour",
                         "account_age_days", "previous_fraud_count"):
                try:
                    clean[key] = max(0, int(value))
                except (TypeError, ValueError):
                    clean[key] = default
                    warnings.append(f"Invalid '{key}', used default value.")
        return clean, warnings

    def _ml_score(self, clean_txn: dict) -> float:
        row = pd.DataFrame([{col: clean_txn.get(col) for col in self.feature_columns}])
        proba = self.pipeline.predict_proba(row)[0, 1]
        return float(proba) * 100

    # -------------------------------------------------------------- Rules --
    def _rule_score(self, clean_txn: dict) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        amount = clean_txn["amount"]
        if amount > 50000:
            score += 25
            reasons.append(f"Transaction amount is unusually high (₹{amount:,.0f})")
        elif amount > 20000:
            score += 12
            reasons.append(f"Transaction amount is above typical range (₹{amount:,.0f})")

        if clean_txn["new_device"]:
            score += 15
            reasons.append("Transaction is from a new/unrecognized device")

        if clean_txn["location_change"]:
            score += 15
            reasons.append("Transaction location differs from customer's usual location")

        tph = clean_txn["transactions_last_hour"]
        if tph >= 3:
            points = min(20, tph * 3)
            score += points
            reasons.append(f"Unusually high transaction velocity ({tph} transactions in the last hour)")

        age = clean_txn["account_age_days"]
        if age < 30:
            score += 15
            reasons.append(f"Very new account ({age} days old)")
        elif age < 90:
            score += 7
            reasons.append(f"Relatively new account ({age} days old)")

        prior = clean_txn["previous_fraud_count"]
        if prior > 0:
            points = min(25, prior * 12)
            score += points
            reasons.append(f"Customer has {prior} prior suspicious transaction(s) on record")

        return min(100.0, score), reasons

    # ------------------------------------------------------------- Public --
    def assess(self, transaction: dict) -> RiskResult:
        clean_txn, warnings = self._sanitize(transaction)

        ml_score = self._ml_score(clean_txn)
        rule_score, rule_reasons = self._rule_score(clean_txn)

        risk_score = (ml_score * ML_WEIGHT) + (rule_score * RULE_WEIGHT)
        risk_score = max(0.0, min(100.0, risk_score))

        if risk_score <= LOW_MAX:
            level = "LOW"
        elif risk_score <= MEDIUM_MAX:
            level = "MEDIUM"
        else:
            level = "HIGH"

        decision = DECISION_BY_LEVEL[level]

        reasons = rule_reasons if rule_reasons else ["No specific risk indicators detected"]
        if ml_score >= self.threshold * 100:
            reasons.insert(0, f"ML model flags this transaction as high-probability fraud ({ml_score:.1f}%)")

        return RiskResult(
            ml_score=ml_score,
            rule_score=rule_score,
            risk_score=risk_score,
            risk_level=level,
            decision=decision,
            reasons=reasons,
            warnings=warnings,
        )


# Module-level singleton, created lazily so importing this file never fails
# just because the model hasn't been trained yet (FastAPI's startup can
# surface a clean error instead of an import-time crash).
_engine: RiskEngine | None = None


def get_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine
