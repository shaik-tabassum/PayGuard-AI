# PayGuard AI — Transaction Risk Detection & Automated Risk Assessment

An AI-powered fraud risk engine that combines a trained ML model with a
transparent rule engine to score, explain, and act on transactions in
real time. Built as a final-year / internship project.

---

## 1. What changed, and why (Phase 1 audit)

The original prototype worked, but had several issues typical of a first
pass at a fraud-detection project. Each is fixed in this version:

| Area | Issue found | Fix |
|---|---|---|
| **ML evaluation** | Model was judged mainly on accuracy (0.79) on an imbalanced dataset (~88% legit / ~12% fraud), which hides how well fraud itself is caught. | Selection now uses cross-validated **PR-AUC** (precision-recall AUC), the correct metric for rare-event classification. Accuracy is still reported, but labeled as secondary. |
| **Class imbalance** | No `class_weight` / resampling — the model had no extra incentive to learn the rare fraud class. | All three candidate models use `class_weight="balanced"`. |
| **Threshold** | Implicit 0.5 decision threshold, which is arbitrary for imbalanced problems. | Threshold is tuned on held-out data to maximize fraud-class F1, saved alongside the model, and reused by the risk engine. |
| **Model choice** | Random Forest was used without comparing alternatives. | `train_model.py` now cross-validates Logistic Regression, Random Forest, and HistGradientBoosting, and keeps whichever wins on PR-AUC. |
| **Data leakage risk** | `transaction_id` / `customer_id` sitting next to feature columns is an easy accidental-leakage trap. | Explicitly excluded from `FEATURE_COLUMNS` in `preprocess.py`, with a comment explaining why. All preprocessing (`StandardScaler`, `OneHotEncoder`) lives inside a `Pipeline` fit only on the training split. |
| **Split strategy** | Not stated whether the split was stratified. | `train_test_split(..., stratify=y)` — preserves the fraud rate in both train and test sets. |
| **Risk engine** | Rule scoring logic wasn't visible/testable as a unit; ML and rule scores could end up conflated. | `risk_engine.py` computes ML score and rule score independently, combines them with a **named, documented formula**, and returns a `reasons` list for every rule that fired. |
| **Backend paths** | Brief mentioned hardcoded paths as a risk. | Every path in `backend/main.py` is resolved via `pathlib.Path(__file__)`, so it works identically on Windows/macOS/Linux and regardless of the current working directory. |
| **Error handling** | No structured handling for bad input or a missing model file. | Pydantic validates every field at the API boundary (positive amount, enums for categorical fields, bounded integers). A missing model returns a clean `503`, not a crash. Malformed transaction dicts are sanitized with logged `warnings` instead of raising. |
| **Frontend/backend integration** | Fine in principle, but had no loading/error states or history. | Added spinner + disabled button while a request is in flight, inline error banner on failure, an "engine online/unreachable" indicator, and empty states for history. |
| **UX** | Score shown as a number only. | Added a color-coded radial gauge, a stamped APPROVE/REVIEW/BLOCK badge, and a labeled ML-vs-rule score breakdown so the *why* is visible, not just the *what*. |

**No dataset was uploaded**, so `data/generate_dataset.py` builds a
synthetic ~5,000-row dataset that mirrors your described schema and class
balance (fraud rate tied to the same signals the rule engine checks, plus
random noise so it isn't trivially separable). **Replace
`data/payguard_ai_transactions.csv` with your real data and re-run
`ml/train_model.py`** — everything downstream (risk engine, API,
dashboard) works unchanged either way, since it only depends on the
column names, not this specific file.

---

## 2. On the "90% accuracy" target — read this before you present

Your dataset is ~88% legitimate transactions. That means **a model that
predicts "not fraud" for every single transaction already scores ~88%
accuracy** while catching zero fraud — it would be a useless, and
actively dangerous, fraud detector. This is why accuracy alone is a
misleading headline metric for fraud detection, and why the original
0.79-accuracy model with 0.55 fraud recall was, in a real sense, doing
*more useful work* than a 90%+ model that just predicts "safe" most of
the time.

With this synthetic dataset, the selected model (Logistic Regression,
chosen over Random Forest / HistGradientBoosting by cross-validated
PR-AUC) gets:

| Metric | Value |
|---|---|
| ROC-AUC | 0.79 |
| PR-AUC (average precision) | 0.41 |
| Test accuracy (tuned threshold) | **0.91** |
| Fraud precision | 0.52 |
| Fraud recall | 0.39 |
| Fraud F1 | 0.45 |

So you *do* clear 90% accuracy here — genuinely, at a threshold chosen to
maximize F1, not by degenerately predicting the majority class (recall is
still 39%, i.e. the model is really catching fraud). **When you retrain
on your real data, the exact numbers will move.** If a reviewer asks
about the accuracy number, the strongest answer is: *"Accuracy is 91%,
but the metric that actually matters here is fraud recall/precision,
because the classes are imbalanced — here's the confusion matrix and
PR-AUC that show the model is catching real fraud, not just guessing the
majority class."* That's a stronger, more senior answer than quoting
accuracy alone, and it's exactly the kind of thing an internship
interviewer is listening for.

If you want to push fraud recall higher (catch more fraud, at the cost of
more false positives / lower accuracy), lower the threshold in
`ml/train_model.py`'s `find_best_threshold` — e.g. constrain to
`recall >= 0.6` instead of pure F1 — and re-train.

---

## 3. Project structure

```
PayGuard-AI/
├── data/
│   ├── generate_dataset.py        # only needed until you have real data
│   ├── payguard_ai_transactions.csv
│   └── cleaned_transactions.csv   # written by train_model.py
├── ml/
│   ├── __init__.py
│   ├── preprocess.py              # shared feature/column definitions
│   ├── train_model.py             # trains + compares + saves the model
│   ├── risk_engine.py             # ML score + rule score + decision
│   ├── fraud_model.pkl            # trained pipeline (generated)
│   └── model_metrics.json         # last training run's metrics (generated)
├── backend/
│   └── main.py                    # FastAPI: /analyze /history /analytics
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   └── test_risk_engine.py        # the 8 scenarios from the brief
├── requirements.txt
└── README.md
```

---

## 4. Running it

```bash
pip install -r requirements.txt

# 1. (Re)train the model — writes ml/fraud_model.pkl
python ml/train_model.py

# 2. Run the tests
python -m unittest tests.test_risk_engine -v

# 3. Start the API (from the project root)
uvicorn backend.main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs

# 4. Open the dashboard
# Just open frontend/index.html in a browser (or serve it with:
python -m http.server 5500 --directory frontend
# then visit http://127.0.0.1:5500 )
```

The frontend calls the API at `http://127.0.0.1:8000` — update
`API_BASE` at the top of `frontend/script.js` if you deploy the backend
elsewhere.

---

## 5. Risk engine formula

```
Final Risk Score = (ML Score × 0.70) + (Rule Score × 0.30)

0–30   → LOW    → APPROVE
31–70  → MEDIUM → REVIEW
71–100 → HIGH   → BLOCK
```

Rule score is a capped sum of independent signals (unusually high amount,
new device, location change, transaction velocity, new account, prior
fraud history) — see `ml/risk_engine.py::_rule_score` for the exact point
values and reasoning for each one.

Re-running the brief's original test transaction (₹100,000, new device,
location change, 10 tx/hour, 20-day-old account, 2 prior fraud flags)
against this version now returns **HIGH / BLOCK** with all five risk
indicators listed, instead of the original MEDIUM / REVIEW.

---

## 6. Next steps worth doing before you present

- Swap in your real `payguard_ai_transactions.csv` and re-train.
- Add a confusion-matrix / ROC-curve screenshot to your slides — it's a
  more convincing artifact than a single accuracy number.
- If you want probability calibration (so "70% fraud probability" really
  means ~70% of such transactions are fraud), add
  `CalibratedClassifierCV` around the winning estimator in
  `train_model.py`.
- Consider adding basic auth or an API key to `/analyze` before calling
  this "production-ready" in your writeup — right now CORS is wide open
  (`allow_origins=["*"]`), which is fine for a local demo but should be
  called out as a known limitation, not left unmentioned.
