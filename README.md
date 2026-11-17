# Predicting Australian Student Visa Outcomes Under Class Imbalance

**A case study in imbalanced classification, applied to a live education/migration policy issue.**

[View the full notebook →](./visa_refusal_classification.ipynb)

---

## The problem

In September 2026, [reporting](https://www.abc.net.au/news/2026-09-09/australian-student-visa-refusal-rate-hits-10-year-high/107112426) based on Department of Home Affairs data showed Australian student visa refusal rates at a **10-year high** — 24.2% nationally, up from 7.9% a decade ago, and as high as **51% for Nepal** and **40% for India** versus **5.8% for China**. Migration agents and lawyers quoted in the coverage described refusal reasoning as inconsistent across case officers.

That's a natural setup for a classification problem — and a genuinely hard one, because "refused" is the **minority class**, at rates ranging from ~6% to ~51% depending on the applicant's country of origin. A model that ignores this imbalance will look accurate on paper while being useless in practice.

**Question:** Can we build a classifier that flags at-risk applications, and do it in a way that's honest about — rather than hidden by — the class imbalance?

## Data

DOHA does not publish application-level data. This project uses a **synthetic dataset calibrated to the published aggregate statistics** (overall and by-country refusal rates) with individual-level features aligned to DOHA's published Genuine Student (GS) assessment criteria: course-career alignment, study gaps, financial evidence, GS statement strength, and education-provider risk tier.

This is a deliberate and disclosed design choice, not a stand-in for real data — the goal is to demonstrate the *methodology* for imbalanced classification on a realistic, well-motivated problem.

## Approach

1. **EDA** to confirm the class imbalance and see how refused vs. approved applications differ on key features
2. **Naive baseline** — plain logistic regression, no imbalance handling
3. **Three imbalance-handling strategies, compared head-to-head:**
   - Class-weighted logistic regression
   - XGBoost with `scale_pos_weight`
   - SMOTE oversampling + logistic regression
4. **Threshold tuning** using the precision-recall curve, rather than the default 0.5 cutoff
5. **Evaluation on metrics that actually matter under imbalance** — recall, precision, F2-score, and PR-AUC on the refused class, never plain accuracy

## Key results

| | Naive baseline | Best model (weighted + threshold-tuned) |
|---|---|---|
| Recall on refusals | 40% | 73% |
| Overall accuracy | 76% *(looks strong, but misleading)* | 70% *(lower, but the model now finds most true refusals)* |

**Class balance and country-level refusal rates:**

![Class balance and refusal rate by country](./eda_class_balance.png)

**How refused applications differ on key features:**

![Feature distributions by outcome](./eda_features.png)

**Choosing a decision threshold instead of defaulting to 0.5:**

![Precision-recall curve and F2-optimal threshold](./threshold_tuning.png)

**Confusion matrix, before and after tuning:**

![Confusion matrices](./confusion_matrices.png)

## So what

- **Accuracy is the wrong headline metric here.** A model predicting "approved" for everyone scores 76–92% accuracy (depending on segment) while catching zero refusals. Any imbalanced classification project should lead with recall/precision/PR-AUC on the minority class instead.
- **Class weighting and threshold tuning together nearly doubled recall** on the refused class (40% → 73%) for a manageable precision trade-off — no exotic techniques required.
- **The strongest predictive features mirror DOHA's own published GS criteria** (statement strength, financial evidence, study gaps, provider risk), which is a useful sanity check on the approach and would be a genuine finding if applied to real application data.
- **Recommended framing for a real deployment:** a pre-submission risk-screening aid for applicants/agents, or a QA layer to flag inconsistent case-officer decisions — always as a flag for human review, not an automated verdict, especially given the lack of appeal rights for offshore applicants.

## Limitations

This is a **methodology demonstration on synthetic data**, not a validated model of real DOHA decisions. Feature-outcome relationships are illustrative of plausible dynamics implied by the GS criteria, not fitted to ground truth. A production version would need a data-sharing partnership with an education provider or migration agency holding real, de-identified outcomes.

## Run it yourself

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter
jupyter notebook visa_refusal_classification.ipynb
```

---

*Real-world statistics cited from: ["Student visa refusals hit 10-year high, shattering dreams," ABC News, 8 Sept 2026](https://www.abc.net.au/news/2026-09-09/australian-student-visa-refusal-rate-hits-10-year-high/107112426), citing Department of Home Affairs data.*
