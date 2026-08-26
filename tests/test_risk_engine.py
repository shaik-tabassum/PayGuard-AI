"""
tests/test_risk_engine.py
--------------------------
Phase 8 test cases. Uses only the standard library's `unittest` so it runs
with nothing beyond what training already requires (no pytest install
needed).

Run from the project root:
    python -m unittest tests.test_risk_engine -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from risk_engine import RiskEngine  # noqa: E402


class PayGuardRiskEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RiskEngine()

    def assess(self, **overrides):
        base = {
            "amount": 1500,
            "payment_method": "UPI",
            "location": "Mumbai",
            "device_type": "Mobile",
            "new_device": 0,
            "location_change": 0,
            "transactions_last_hour": 1,
            "account_age_days": 900,
            "previous_fraud_count": 0,
        }
        base.update(overrides)
        return self.engine.assess(base)

    # 1. Safe transaction --------------------------------------------------
    def test_01_safe_transaction_is_low_and_approved(self):
        result = self.assess()
        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(result.decision, "APPROVE")

    # 2. High-value transaction ---------------------------------------------
    def test_02_high_value_transaction_raises_risk(self):
        low = self.assess()
        high_value = self.assess(amount=250000)
        self.assertGreater(high_value.risk_score, low.risk_score)
        self.assertIn("unusually high", " ".join(high_value.reasons).lower())

    # 3. New device -----------------------------------------------------------
    def test_03_new_device_raises_risk_and_is_reported(self):
        baseline = self.assess()
        flagged = self.assess(new_device=1)
        self.assertGreater(flagged.risk_score, baseline.risk_score)
        self.assertTrue(any("device" in r.lower() for r in flagged.reasons))

    # 4. Location change -------------------------------------------------------
    def test_04_location_change_raises_risk_and_is_reported(self):
        baseline = self.assess()
        flagged = self.assess(location_change=1)
        self.assertGreater(flagged.risk_score, baseline.risk_score)
        self.assertTrue(any("location" in r.lower() for r in flagged.reasons))

    # 5. Multiple transactions in a short window --------------------------------
    def test_05_high_velocity_raises_risk(self):
        baseline = self.assess()
        flagged = self.assess(transactions_last_hour=12)
        self.assertGreater(flagged.risk_score, baseline.risk_score)
        self.assertTrue(any("velocity" in r.lower() for r in flagged.reasons))

    # 6. Previous fraud history --------------------------------------------------
    def test_06_previous_fraud_history_raises_risk(self):
        baseline = self.assess()
        flagged = self.assess(previous_fraud_count=3)
        self.assertGreater(flagged.risk_score, baseline.risk_score)
        self.assertTrue(any("prior suspicious" in r.lower() for r in flagged.reasons))

    # 7. Combination of all risk indicators -> should escalate to HIGH/BLOCK ----
    def test_07_combination_of_indicators_is_high_and_blocked(self):
        result = self.assess(
            amount=100000,
            new_device=1,
            location_change=1,
            transactions_last_hour=10,
            account_age_days=20,
            previous_fraud_count=2,
        )
        self.assertEqual(result.risk_level, "HIGH")
        self.assertEqual(result.decision, "BLOCK")
        self.assertGreaterEqual(len(result.reasons), 4)

    # 8. Missing / invalid values are handled safely, never crash ---------------
    def test_08_missing_and_invalid_fields_degrade_gracefully(self):
        result = self.engine.assess({
            "amount": "not-a-number",
            "payment_method": "Card",
            # location, device_type intentionally omitted
            "new_device": 1,
            "transactions_last_hour": -5,  # invalid: negative
            "account_age_days": 40,
            "previous_fraud_count": 0,
        })
        self.assertIn(result.risk_level, {"LOW", "MEDIUM", "HIGH"})
        self.assertTrue(len(result.warnings) > 0)


if __name__ == "__main__":
    unittest.main()
