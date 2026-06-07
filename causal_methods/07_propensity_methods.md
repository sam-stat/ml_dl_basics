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
  - **Balancing theorem (Rosenbaum & Rubin, 1983).** Unconfoundedness on $X$ implies it on the scalar
    $e(X)$, because $X \perp\!\!\!\perp T \mid e(X)$ — collapsing the high-dimensional vector into one
    scalar loses nothing for confounding adjustment, provided $\hat e(x)$ actually balances $X$ in practice.

---

## The PSM algorithm

Matching rebuilds an approximate experiment by giving every treated unit a statistical twin.

- **① Select covariates $X$.** Include every confounder (affects both treatment and outcome).
  **Exclude post-treatment variables** — anything *caused by* the treatment biases the score.
- **② Fit the propensity model.** Binary classifier with target $T$, features $X$ (logistic regression
  classically; tree ensembles when flexible). Predict $\hat e(x)$ for every unit.
- **③ Choose the fitting scheme — same data vs K-fold cross-fitting.**
  - *Same data.* Defensible for classical logistic-regression PSM — $Y$ never enters the propensity
    model, so overfitting $T$ only yields a noisier matching key, and success is judged by balance on
    this same sample.
  - *Issues with same-data fitting*, especially under flexible models (GBM / RF / NN):
    - the model can *perfectly separate* the arms in-sample, producing a spurious bimodal-at-0/1
      $\hat e(x)$ histogram — overfitting masquerading as a positivity violation;
    - weights blow up or most of the sample is discarded under caliper;
    - balance diagnostics on the training sample stop being a reliable check.
  - *K-fold cross-fitting* fixes this. Split the data into $K$ folds (typically $K = 5$ or $10$); for
    each fold $k$, fit the model on the other $K - 1$ folds and predict $\hat e(x)$ on fold $k$
    (out-of-fold); stack so every unit has a score from a model that never trained on it. Use these
    out-of-fold scores for matching / weighting downstream — no separate holdout needed.
- **④ Check overlap.** Plot $\hat e(x)$ by group; keep the common-support region.
- **⑤ Match.** Nearest-neighbour on $\hat e(x)$. Parameters:
  - *with / without replacement* — may one control serve several treated (usually without);
  - *caliper* — maximum allowed score gap; a treated unit with no control inside it is dropped;
  - variants: $k{:}1$, kernel, stratification, full / optimal matching.
- **⑥ Assess balance.** Standardized Mean Difference (SMD) per covariate on the matched sample; **SMD < 0.1**
  is the usual threshold. If it fails, revise $X$ / model / caliper and re-match.
- **⑦ Estimate the effect** as a mean difference on the matched sample.

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

The failure is local and concrete: in some region of $\hat e(x)$, a treated unit has no comparable
control to borrow $\hat Y(0)$ from (or a control has no comparable treated to borrow $\hat Y(1)$ from),
so there is no counterfactual to impute.

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

- **ATE is hit harder than ATT.** ATT only needs controls where treated units sit; ATE needs matches in
  *both* directions, so the abundant-control / few-treated tail also breaks — no treated units to impute
  $\hat Y(1)$ for those controls.
- **Deterministic assignment = total failure.** If treatment is a deterministic rule on variables that
  are themselves in $X$, then $\hat e(x) \in \{0, 1\}$: the two arms live on disjoint points and the
  effect is **not identified** — a property of the data, not the algorithm.
- **ATT can still fail, but narrowly** — treated units in regions with no comparable controls force
  either bad far matches (bias) or caliper drops; after drops you estimate ATT on the matched subset,
  not the full ATT, so report the drop count.
- **Fixes when overlap is poor.** Restrict to the common-support / overlap subpopulation (a redefined
  estimand — the ATO); inject randomization or find exogenous variation in assignment; use trimming or
  overlap weights (below).

> **Key idea:** if **positivity** or **unconfoundedness** breaks, no method recovers the effect — PSM,
> IPW, regression, AIPW, and causal forests all fail in the same region for the same reason. Plot the
> $\hat e(x)$ histograms first; bimodal-at-0-and-1 means the effect is not identified.

---

## Inverse Probability Weighting (IPW / IPTW)

Instead of discarding unmatched units, reweight all of them so the weighted arms resemble the target
population — a synthetic randomized experiment.

- **Weights.**
$$w_i = \begin{cases} 1 / \hat e(x_i), & T_i = 1 \\[4pt] 1 / (1 - \hat e(x_i)), & T_i = 0 \end{cases}$$
- **Why it works.**
  - Upweight the **under-represented** units in each arm so each arm matches the population on $X$.
  - A unit with low $e(x)$ observed as *treated* is rare relative to how often that profile appears in
    the population → its weight $1/e$ is large, letting it stand in for the similar units who went untreated.
  - The same unit observed as *control* is abundant in that region → it gets a weight near 1, no
    amplification needed.
  - After weighting, the treated and control arms have matching covariate distributions, so the
    weighted mean difference behaves like a synthetic randomized experiment.
  - This is the same inverse-probability correction Horvitz & Thompson (1952) used for
    unequal-probability survey sampling.
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
  - **Overlap weights** (Li, Morgan, Zaslavsky 2018) — $1 - \hat e(x)$ for treated, $\hat e(x)$ for
    controls; bounded in $[0, 1]$, cannot explode, and implicitly target the overlap population.

> **Key idea:** matching is a 0/1 special case of weighting (neighbourhood membership). IPW is the more
> general lens — reach for matching when you want an auditable cohort story and the ATT, and for IPW
> (with trimming or overlap weights) when you want the ATE on all the data.
