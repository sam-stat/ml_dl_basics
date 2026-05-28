# Propensity-Score Methods: Matching & Weighting

Observational data has no randomization: units self-select into treatment on covariates $X$, so a raw
treated-minus-control gap mixes the treatment effect with **selection bias**. The fix is to compare
like with like — but exact matching on a high-dimensional $X$ is hopeless (no two units share a
50-dimensional profile: the **curse of dimensionality**). Propensity-score methods collapse $X$ into a
single scalar and adjust on that.

- **Propensity score.** $e(x) = P(T = 1 \mid X = x)$ — the probability of being treated given covariates.
- **Two methods, one score.**
  - **PSM (matching)** — pair each treated unit with control unit(s) of near-identical $e(x)$, discard
    the rest, take a mean difference. Natural for **ATT**.
  - **IPW / IPTW (weighting)** — keep every unit, weight by the inverse of its treatment probability to
    build a balanced pseudo-population. Natural for **ATE**.
- **Shared assumptions** (both inherit [ignorability](02_estimands.md)):
  - **Unconfoundedness** — $(Y(1), Y(0)) \perp\!\!\!\perp T \mid X$: $X$ captures everything driving both
    treatment and outcome.
  - **Positivity (overlap)** — $0 < e(x) < 1$ for every $x$: every unit could in principle land in either arm.

---

## Why one scalar is enough — the balancing theorem

- **Theorem (Rosenbaum & Rubin, 1983).** If unconfoundedness holds on the full vector $X$, it also holds
  on the scalar score:
$$(Y(1), Y(0)) \perp\!\!\!\perp T \mid X \quad\Longrightarrow\quad (Y(1), Y(0)) \perp\!\!\!\perp T \mid e(X)$$
- **Why it makes sense.** $e(X)$ is a **balancing score**: among units that share the same $e(X)$, the
  *distribution of $X$ is identical* across treated and control — $X \perp\!\!\!\perp T \mid e(X)$. Two
  units can reach $e = 0.3$ via different covariate profiles, but on average treated and control units at
  a given score carry the same covariates, so conditioning on the one number removes the same confounding
  as conditioning on the whole vector.
- **What it buys.** A 1-D sufficient statistic for confounding adjustment — match or weight on a scalar
  instead of an intractable high-dimensional vector.
- **What it does *not* relax.**
  - Holds for the **true** $e(x)$; in practice you fit $\hat e(x)$, so a misspecified model breaks balance
    → **balance diagnostics are mandatory**.
  - Says nothing about **unobserved** confounders — compressing a flawed adjustment set into a scalar keeps the flaw.

> **Key idea:** the propensity score is a dimensionality-reduction trick *conditional on*
> selection-on-observables. It makes the estimator tractable; it does not weaken the identifying assumption.

---

## The PSM algorithm

Matching rebuilds an approximate experiment by giving every treated unit a statistical twin.

- **① Select covariates $X$.** Include every confounder (affects both treatment and outcome).
  **Exclude post-treatment variables** — anything *caused by* the treatment biases the score.
- **② Fit the propensity model.** Binary classifier with target $T$, features $X$ (logistic regression
  classically; tree ensembles when flexible). Predict $\hat e(x)$ for every unit.
- **③ Check overlap.** Plot $\hat e(x)$ by group; keep the common-support region.
- **④ Match.** Nearest-neighbour on $\hat e(x)$. Parameters:
  - *with / without replacement* — may one control serve several treated (usually without);
  - *caliper* — maximum allowed score gap; a treated unit with no control inside it is dropped;
  - variants: $k{:}1$, kernel, stratification, full / optimal matching.
- **⑤ Assess balance.** Standardized Mean Difference (SMD) per covariate on the matched sample; **SMD < 0.1**
  is the usual threshold. If it fails, revise $X$ / model / caliper and re-match.
- **⑥ Estimate the effect** as a mean difference on the matched sample.

---

## Computing the ATT

The treated group *is* the target population; controls only supply the missing $Y_i(0)$.

- Loop over each treated unit $i$ ($T_i = 1$); find its $k$ nearest controls $\mathcal{M}(i)$ by $\hat e(x)$.
- Impute the counterfactual as the matched-control mean; the per-unit effect is observed minus imputed.
- Average over the treated:
$$\widehat{\text{ATT}} = \frac{1}{N_1} \sum_{i:\,T_i = 1} \Big( Y_i - \frac{1}{k} \sum_{j \in \mathcal{M}(i)} Y_j \Big)$$
- Only the **treated side** needs control coverage — you never search for matches of untreated units.

---

## Computing the ATE

ATE asks the effect over *everyone*, so both arms need an imputed counterfactual → the procedure becomes symmetric.

- **Forward pass.** For each treated $i$, match controls → impute $\hat Y_i(0)$.
- **Reverse pass.** For each control $j$, match treated → impute $\hat Y_j(1)$.
- **Per-unit effects, then average over all $N$:**
$$\hat\tau_i = \begin{cases} Y_i - \hat Y_i(0), & T_i = 1 \\[4pt] \hat Y_i(1) - Y_i, & T_i = 0 \end{cases} \qquad \widehat{\text{ATE}} = \frac{1}{N} \sum_{i=1}^{N} \hat\tau_i$$
- Now **both sides** need coverage — controls for the treated *and* treated for the controls.

---

## Positivity: where ATT and ATE break

Positivity is the assumption most often silently violated — and the violation sinks every method, not just PSM.

```
Density
│     ╱╲  controls
│    ╱   ╲           ╱╲  treated
│   ╱     ╲         ╱   ╲
│  ╱       ╲       ╱     ╲
│ ╱         ╲    ╱        ╲
└────────────────────────────→ e(x)
0                           1
```

- **Overlap region.** The treated and control $\hat e(x)$ distributions overlap in the middle and thin
  out at opposite tails.
- **ATE is hit harder than ATT.**
  - ATT needs controls only where treated units sit — the abundant-control / few-treated tail is
    irrelevant (no treated units to match there).
  - ATE needs matches both ways, so that same tail now demands treated matches for abundant controls —
    imputing the outcome of "treating a unit that is essentially never treated" is **extrapolation, not matching**.
- **Deterministic assignment = total positivity failure.** If treatment is a deterministic rule on
  variables that are themselves in $X$, then $\hat e(x) \in \{0, 1\}$: the two arms live on disjoint
  points, there is **no overlap**, and the effect is **not identified** — a property of the data, not the algorithm.
- **The absence of overlap breaks every method, not only PSM** — it just surfaces differently:
  - **PSM** — no controls share the treated scores; matching is undefined.
  - **IPW** — $1/\hat e(x)$ or $1/(1 - \hat e(x))$ → division by zero, weights → $\infty$.
  - **Regression / S-learner** — one arm is empty in that region; the model only extrapolates from
    functional-form assumptions.
  - **AIPW / TMLE** — both nuisance models fail in the same region; double robustness guards against
    *misspecification when positivity holds*, not against its violation.
  - **Causal forests / meta-learners** — no within-neighbourhood treatment variation → nothing to split
    on, no counterfactual signal.
- **ATT can still fail, but narrowly** — treated units in regions with no comparable controls force
  either bad far matches (bias) or caliper drops. After drops you estimate ATT on the *matched subset*,
  not the full ATT; report how many treated units were removed.
- **Fixes when overlap is poor.**
  - Restrict to the common-support / overlap subpopulation (a redefined estimand — the ATO).
  - Inject randomization (a small holdout) or find exogenous variation in assignment.
  - Use stabilized IPW or overlap weights (below).
  - Don't silently drop perfectly-separating deterministic-rule covariates — that trades a positivity
    violation for a confounding one.

> **Key idea:** plot the $\hat e(x)$ histograms *first*. A bimodal-at-0-and-1 shape means the effect is
> not identified — no method recovers it.

---

## Data for the propensity model: same sample vs cross-fitting

The propensity model is a **nuisance** function, not the final answer, so its train/test logic differs from predictive ML.

- **Classical PSM — the same data is fine.**
  - The model predicts $T$ from $X$; the outcome $Y$ never enters it, so overfitting $T$ only yields a
    noisier *matching key*, not bias in the treatment–outcome link.
  - Success is judged by **balance on this sample**, not out-of-sample accuracy.
  - Abadie & Imbens (2016): using the *estimated* score can even *lower* the matching estimator's
    variance versus the true score.
- **When same-data breaks — flexible, high-capacity models.** GBM / RF / NN can find an $\hat e(x)$ that
  *perfectly separates* the arms in-sample even when true overlap is fine → a bimodal-at-0/1 histogram.
  This is **overfitting masquerading as a positivity violation**; weights explode or most of the sample
  gets discarded.
- **Fix — cross-fitting** (DML-style; Chernozhukov et al. 2018).
  - Split into $K$ folds (typically $K = 5$ or $10$).
  - For each fold $k$: fit the model on the other $K - 1$ folds, predict $\hat e(x)$ on fold $k$ (**out-of-fold**).
  - Stack predictions → every unit has a score from a model that never saw it during training.
  - Use these out-of-fold scores downstream. This restores $\sqrt{n}$-consistency and asymptotic
    normality even with ML nuisances; repeat over several splits to cut split-induced variance.
  - **Cross-fitting replaces the need for a separate holdout** — the $K$-fold structure supplies the independence.
- **Decision rule.**

| Use same-data when | Use cross-fitting when |
|--------------------|------------------------|
| logistic regression, modest feature set | GBM / RF / neural-net propensity model |
| classical PSM with balance checks | DML, AIPW, TMLE, any doubly-robust method |
| large $n$ vs dimensionality, sane histogram | CATE via meta-learners (orthogonality needs it) |
| | high-dimensional $X$, tight $n/p$, high-stakes results |

- An **outcome model** $\hat\mu_t(x) = E[Y \mid X, T]$ (for AIPW / meta-learners) touches $Y$ directly →
  cross-fitting is effectively mandatory there.

---

## Inverse Probability Weighting (IPW / IPTW)

Instead of discarding unmatched units, reweight all of them so the weighted arms resemble the target
population — a synthetic randomized experiment.

- **Weights.**
$$w_i = \begin{cases} 1 / \hat e(x_i), & T_i = 1 \\[4pt] 1 / (1 - \hat e(x_i)), & T_i = 0 \end{cases}$$
- **Why it works.** Upweight the under-represented in each arm. A unit with low $e(x)$ observed as
  *treated* is rare relative to how often that profile appears in the population, so its weight $1/e$ is
  large and lets it stand in for the similar units who went untreated; the same unit observed as
  *control* is abundant and gets a weight near 1. After weighting, each arm's covariate distribution
  matches the population's. (The same inverse-probability correction Horvitz & Thompson (1952) used for
  unequal-probability survey sampling.)
- **ATE estimator** — a weighted mean difference; no matching, no caliper:
$$\widehat{\text{ATE}}_{\text{IPW}} = \frac{1}{N} \sum_{i=1}^{N} \frac{T_i\, Y_i}{\hat e(x_i)} \;-\; \frac{1}{N} \sum_{i=1}^{N} \frac{(1 - T_i)\, Y_i}{1 - \hat e(x_i)}$$
- **ATT by reweighting** — treated keep weight 1 (already the target); controls are tilted to the treated
  covariate distribution:
$$w_i^{\text{ATT}} = \begin{cases} 1, & T_i = 1 \\[4pt] \dfrac{\hat e(x_i)}{1 - \hat e(x_i)}, & T_i = 0 \end{cases}$$
- **ATC** flips the ATT logic (reweight treated to look like the controls).

---

## IPW vs PSM, and how to make IPW robust

Both use the same score; the trade-off is data efficiency and a clean ATE versus interpretability and robustness.

| | PSM (matching) | IPW (weighting) |
|---|----------------|-----------------|
| Data use | discards unmatched units | keeps all units |
| Natural estimand | ATT | ATE (one formula, no reverse pass) |
| Continuous treatment | awkward | natural (generalized scores) |
| Interpretability | high ("matched cohort") | lower ("reciprocal weights") |
| Sensitivity to $\hat e$ error | forgiving (small → small) | amplified through the reciprocal, esp. in tails |
| Main failure mode | dropped treated under caliper | weight explosion near $e \to 0, 1$ |

- **IPW's Achilles heel — weight explosion.** As $\hat e(x) \to 0$ or $1$, weights blow up and a single
  unit can dominate the estimate. Remedies:
  - **Trimming / truncation** — cap weights (e.g. 1st/99th percentile, or a fixed ceiling). Small bias,
    large variance reduction.
  - **Stabilized weights** (Robins et al. 2000) — multiply by the marginal treatment probability so weights cluster near 1:
$$w_i^{\text{stab}} = \begin{cases} P(T = 1) / \hat e(x_i), & T_i = 1 \\[4pt] P(T = 0) / (1 - \hat e(x_i)), & T_i = 0 \end{cases}$$
  - **Overlap weights** (Li, Morgan, Zaslavsky 2018) — $1 - \hat e(x)$ for treated, $\hat e(x)$ for
    controls; bounded in $[0, 1]$, cannot explode, and implicitly target the overlap population.
- **Doubly-robust upgrade — AIPW.** Combine weighting with an outcome model $\hat\mu_t(x)$; consistent if
  *either* the propensity *or* the outcome model is correct:
$$\widehat{\text{ATE}}_{\text{AIPW}} = \frac{1}{N} \sum_{i=1}^{N} \Big[ \hat\mu_1(x_i) - \hat\mu_0(x_i) + \frac{T_i\,(Y_i - \hat\mu_1(x_i))}{\hat e(x_i)} - \frac{(1 - T_i)\,(Y_i - \hat\mu_0(x_i))}{1 - \hat e(x_i)} \Big]$$
  With cross-fitting this is the DML / AIPW workhorse behind EconML and DoubleML (see [DML](06_double_machine_learning.md)).

> **Key idea:** matching is a 0/1 special case of weighting (neighbourhood membership). IPW is the more
> general lens — reach for matching when you want an auditable cohort story and the ATT, and for IPW
> (stabilized / overlap, or AIPW) when you want the ATE on all the data.
