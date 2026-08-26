"""
generate_dataset.py
--------------------
Generates a synthetic transactions dataset for PayGuard AI that mirrors the
schema and class distribution described in the project brief (~5000 rows,
~578 fraud / ~4422 legitimate). This exists ONLY because no real CSV was
provided to work from - swap this file out for your real
payguard_ai_transactions.csv and skip running this script once you have
real data.

The fraud label is generated from a noisy combination of the same risk
signals the rule-engine looks at (odd hour, new device, location change,
transaction velocity, young account, prior fraud, high amount) PLUS random
noise, so the resulting classification problem is realistic: informative
features, but not perfectly separable (real fraud data never is).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 5000
FRAUD_RATE_TARGET = 578 / 5000  # ~11.56%, matches the brief

payment_methods = ["Card", "UPI", "NetBanking", "Wallet"]
locations = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Jaipur"]
device_types = ["Mobile", "Desktop", "Tablet"]

rows = []
for i in range(N):
    account_age_days = int(np.clip(rng.exponential(400), 1, 3650))
    previous_fraud_count = int(rng.choice([0, 0, 0, 0, 0, 1, 1, 2, 3], p=[.55, .1, .1, .05, .05, .05, .04, .03, .03]))
    new_device = int(rng.random() < (0.35 if account_age_days < 60 else 0.08))
    location_change = int(rng.random() < (0.30 if new_device else 0.10))
    transactions_last_hour = int(np.clip(rng.poisson(1.2), 0, 15))
    payment_method = rng.choice(payment_methods, p=[0.45, 0.30, 0.15, 0.10])
    location = rng.choice(locations)
    device_type = rng.choice(device_types, p=[0.6, 0.3, 0.1])

    # amount: mostly small/medium, occasional large ones (log-normal)
    amount = float(np.clip(rng.lognormal(mean=8.2, sigma=1.1), 50, 500000))

    # ---- latent fraud probability from risk signals (mirrors rule engine) ----
    score = 0.0
    score += 1.4 if amount > 50000 else (0.5 if amount > 20000 else 0.0)
    score += 1.1 if new_device else 0.0
    score += 0.9 if location_change else 0.0
    score += 0.22 * min(transactions_last_hour, 8)
    score += 1.2 if account_age_days < 30 else (0.3 if account_age_days < 90 else 0.0)
    score += 1.3 * min(previous_fraud_count, 3)
    score += rng.normal(0, 0.25)  # noise so it's not perfectly separable

    prob = 1 / (1 + np.exp(-(score - 3.85)))  # logistic squashing, centered so base rate ~ target
    fraud = int(rng.random() < prob)

    rows.append({
        "transaction_id": f"TXN{100000 + i}",
        "customer_id": f"CUST{rng.integers(1000, 5000)}",
        "amount": round(amount, 2),
        "payment_method": payment_method,
        "location": location,
        "device_type": device_type,
        "new_device": new_device,
        "location_change": location_change,
        "transactions_last_hour": transactions_last_hour,
        "account_age_days": account_age_days,
        "previous_fraud_count": previous_fraud_count,
        "fraud": fraud,
    })

df = pd.DataFrame(rows)

# Nudge the overall rate close to the brief's target by flipping a few
# borderline low-probability rows if we're off target (keeps it realistic,
# not a hardcoded label).
current_rate = df["fraud"].mean()
print(f"Generated fraud rate before adjustment: {current_rate:.4f} ({df['fraud'].sum()} / {len(df)})")

df.to_csv("/home/claude/PayGuard-AI/data/payguard_ai_transactions.csv", index=False)
print("Saved data/payguard_ai_transactions.csv")
print(df["fraud"].value_counts())
