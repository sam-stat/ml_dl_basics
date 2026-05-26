# Causal Methods — Notes Index

> A structured reference for causal inference and causal machine learning — from the
> fundamental problem, through estimands and the method taxonomy, to the meta-learner
> and double-machine-learning estimators used to personalize treatment effects.
> All diagrams are SVG files in `diagrams/` — rendered natively on GitHub.

---

## Contents

| # | Topic | File |
|---|-------|------|
| 1 | Foundations — correlation, causation, the fundamental problem | [01_foundations.md](01_foundations.md) |
| 2 | Estimands — the two defining frameworks, then ATE / CATE / ATT (and CATE → uplift) | [02_estimands.md](02_estimands.md) |
| 3 | The methods taxonomy — `<type>: <method>` | [03_methods_taxonomy.md](03_methods_taxonomy.md) |
| 4 | Evaluation — why ROC/R² fail, cumulative-gain & Qini | [04_evaluation.md](04_evaluation.md) |
| 5 | Causal ML: Meta-Learners (S / T / X) | [05_meta_learners.md](05_meta_learners.md) |
| 6 | Causal ML: Double Machine Learning (DML) | [06_double_machine_learning.md](06_double_machine_learning.md) |

---

## How to read these notes (the learning trajectory)

Estimation is the math you do *after* you have proven the math is worth doing. A robust path:

1. **Foundations & identification** — *can* the effect be isolated from this data at all? (file 1)
2. **Frameworks & estimand** — the language for effects, then name the exact number: ATE? CATE? ATT? (file 2)
3. **Pick a method** — match the tool to the data you have. (file 3)
4. **Evaluate** — prove the estimate is real, despite never seeing the counterfactual. (file 4)
5. **Estimate at scale** — meta-learners and DML for per-user effects. (files 5–6)
6. **Refute & deploy** — stress-test assumptions, then act on the scores. (future scope, below)

---

## Notation used across all notes

| Symbol | Meaning |
|--------|---------|
| $Y$ | observed outcome |
| $Y(1),\ Y(0)$ | potential outcomes — what would happen *with* / *without* treatment |
| $T \in \{0, 1\}$ | treatment indicator (1 = treated) |
| $X$ | covariates / user feature vector |
| $\tau(x)$ | CATE — conditional average treatment effect at $X = x$ |
| $\text{ATE},\ \text{CATE},\ \text{ATT}$ | the three estimands (file 2) |
| $e(x) = P(T=1 \mid X=x)$ | propensity score |
| $\mu_1(x),\ \mu_0(x)$ | outcome models fit on treated / control |
| $D_1,\ D_0$ | imputed individual effects in the X-learner |
| $\tilde{Y},\ \tilde{T}$ | residuals $Y - E[Y\mid X]$, $T - E[T\mid X]$ (DML) |
| $\perp\!\!\!\perp$ | statistical independence |

---

## Planned / future notes (scope to extend)

These are flagged by the source material but not yet written. Each can become a new
`NN_*.md` file following the same `<type>: <method>` heading convention.

- **`Foundational: Identification & Causal Discovery`** — DAGs, backdoor criterion, confounders vs colliders, the PC / FCI discovery algorithms.
- **`Foundational: do-calculus`** — formalizing interventions $do(X=x)$.
- **`Quasi-Experimental: Propensity methods (deep dive)`** — matching, IPW, propensity *binning* as a bias purifier.
- **`Validation: Refutation & sensitivity analysis`** — placebo tests, dummy confounders, bounding unobserved confounding.
- **`Causal ML: Causal feature importance`** — Causal SHAP, uplift trees (*why* a user is persuadable, not just whether).
- **`Causal ML: Continuous / dosage treatments`** — dose-response curves, diminishing returns.
- **`Deployment: Optimization & policy learning`** — budgeted assignment as a knapsack / BILP problem, the perpetual random holdout for model degradation.
- **An executable companion notebook** for DML / meta-learners (EconML, CausalML, DoWhy).

---

## References

From the source material:

- Pearl, J. (2009). *Causality*. Cambridge University Press.
- Wager, S., & Athey, S. (2015). *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*. arXiv:1510.04342.
- Yao, L., Chu, Z., Li, S., et al. (2021). *A Survey on Causal Inference*. ACM TKDD, 15, 1–46.
- Bo, N., Wei, Y., Zeng, L., Kang, C., & Ding, Y. (2024). *A Meta-Learner Framework to Estimate Individualized Treatment Effects for Survival Outcomes*. Journal of Data Science, 505–523.

Canonical references for the deep-dive methods (standard sources, added for completeness):

- Künzel, S., Sekhon, J., Bickel, P., & Yu, B. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning* (S/T/X-learners). PNAS, 116(10).
- Chernozhukov, V., et al. (2018). *Double/Debiased Machine Learning for Treatment and Structural Parameters*. The Econometrics Journal, 21(1).
