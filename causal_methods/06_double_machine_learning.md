# Causal ML: Double Machine Learning (DML)

[Meta-learners](05_meta_learners.md) assume the effect is already *identified* (ignorability holds).
**Double Machine Learning (DML)** — also the **R-learner** in its per-user / Conditional Average
Treatment Effect (CATE) form — is built for the messier reality: **observational data with many
confounders**, where treatment was assigned by some unknown, biased process. The trick: use ML to
*predict the confounding away* from both treatment and outcome, then estimate the effect from what's
left — the **residuals**, or "surprises."

![DML — partial out, then regress](diagrams/dml_residualization.svg)

DML descends from a classic econometrics result, the **Frisch–Waugh–Lovell (FWL) theorem**,
modernized to allow arbitrary ML nuisance models by Chernozhukov et al. (2018).

---

## The problem: weigh the captain, not the ship

If conversion is 90% baseline (credit score, app history) and 10% treatment, a model predicting
conversion directly spends almost all its capacity on the baseline — the causal signal is a rounding
error. DML **removes the baseline first** so the model only ever sees the part the baseline *can't*
explain.

---

## Stage 1 — residualize (partial out the confounders $X$)

Fit two **nuisance models** with any flexible ML:

- **Outcome model:** $m(X) = E[\,Y \mid X\,]$
- **Treatment model:** $e(X) = E[\,T \mid X\,]$ (the propensity)

Take the **residuals** — the part $X$ failed to explain:

$$\tilde{Y} = Y - m(X) \qquad\qquad \tilde{T} = T - e(X)$$

- $\tilde{Y}$ = **outcome surprise** (converted more/less than the profile predicted).
- $\tilde{T}$ = **treatment surprise** (treated more/less than the assignment rules predicted).
- Confounding that operated *through* $X$ is now gone from both.

---

## Stage 2 — regress surprise on surprise

$$\hat{\theta} = \arg\min_{\theta} \sum_i \big(\tilde{Y}_i - \theta\,\tilde{T}_i\big)^2$$

The slope **is** the causal effect: *when a user was unexpectedly treated, were they also unexpectedly
converted?* Both axes are residuals, so the relationship between them cannot be explained by $X$.

> **Key idea:** a residual is "what's left after subtracting the expectation." Regressing the outcome
> residual on the treatment residual asks the causal question while holding *everything predictable
> from $X$* fixed.

---

## Why it works: orthogonality

DML tolerates *imperfect* nuisance models because of **Neyman orthogonality**: the estimating equation
is built so small errors in $m(X)$ or $e(X)$ have only a *second-order* effect on $\hat\theta$ — they
roughly cancel rather than compound. Hence *double*: get *both* nuisance models *approximately* right
and residualization protects the estimate.

---

## Cross-fitting (don't skip it)

Flexible models overfit, and overfit residuals are **biased toward zero**. **Cross-fitting** fixes it:
split into folds, fit $m$ and $e$ on one part, compute residuals on the *held-out* part, rotate. Each
residual comes from models that never saw that row → the overfitting bias cancels. Skip it and DML
quietly underestimates the effect.

---

## From ATE to CATE: the R-learner

The slope above is one number (ATE-style). For a *per-user* effect $\tau(x)$, make the slope a
**function of features**:

$$\hat{\tau} = \arg\min_{\tau} \sum_i \tilde{T}_i^{\,2}\left(\frac{\tilde{Y}_i}{\tilde{T}_i} - \tau(x_i)\right)^2$$

This is the **R-learner** — the same residual-on-residual idea, but the target is a CATE function. It
is the fourth meta-learner: the one that *also* removes confounding rather than assuming it away.

---

## Where DML meets propensity binning

$e(X)$ is exactly the **propensity score**. An older relative — **propensity binning / matching** —
sorts users into bins of similar $e(X)$ and compares treated vs control *within* each bin ("twins").
Same disease (selection bias), same quantity: DML *subtracts* the propensity (residualizes); matching
*stratifies* on it.

---

## DML vs meta-learners — when to use which

| | Meta-learners (S/T/X) | DML / R-learner |
|---|----------------------|------------------|
| **Effect identified?** | yes — needs ignorability up front | built to *remove* observed confounding |
| **Best data** | RCT or clean adjustment set | observational, **many** confounders |
| **Mechanism** | model outcomes, then difference / impute | residualize $Y$ and $T$, then regress |
| **Key safeguard** | propensity weighting (X-learner) | cross-fitting + orthogonality |
| **Main risk** | confounding it can't see | weak nuisance models, no cross-fitting |

> **Key idea:** reach for **DML** when treatment was assigned non-randomly on observed features and a
> strong baseline would swamp the effect; reach for **meta-learners** when the effect is already
> identified (an experiment) and you just want to personalize it. Both live in **EconML** and
> **CausalML**.
