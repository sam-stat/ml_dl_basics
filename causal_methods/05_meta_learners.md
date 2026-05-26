# Causal ML: Meta-Learners

A standard ML model predicts a *single* outcome — *will this user convert?* It has no native notion of
a treatment effect, because an effect is a **difference between two parallel worlds** and the model
sees only one. A **meta-learner** is a recipe that orchestrates ordinary models (XGBoost, LightGBM,
logistic regression) to estimate the **Conditional Average Treatment Effect (CATE)**
$\tau(x) = E[Y(1) - Y(0)\mid X=x]$ anyway. "Meta" = *a learner about learners*: no new gradient or
split rule, just standard learners arranged in stages.

Assumes ignorability (an RCT or a credible adjustment set) — meta-learners *personalize* an
already-identified effect; they don't fix confounding (that's [DML](06_double_machine_learning.md)).

![Meta-learners — S, T, X](diagrams/meta_learners.svg)

---

## S-Learner (single model)

One model on all the data, with treatment $T$ as just another feature:

$$\mu(x, t) = E[\,Y \mid X = x, T = t\,] \qquad \hat{\tau}(x) = \mu(x, 1) - \mu(x, 0)$$

- **Pro.** Simplest; one model; stable.
- **Con.** With hundreds of features and one $T$ column, a regularized learner may **split $T$ away**
  → $\hat{\tau}(x) \to 0$ for everyone. The treatment signal drowns.

---

## T-Learner (two models)

Separate the data; train two independent models:

$$\mu_1(x) = E[Y\mid x, T=1], \quad \mu_0(x) = E[Y\mid x, T=0] \qquad \hat{\tau}(x) = \mu_1(x) - \mu_0(x)$$

- **Pro.** $T$ can never be regularized away — it picks the model.
- **Con.** The models never talk; each minimizes *its own* error, so independent errors **add** on
  subtraction → high variance. Worse under imbalance: a tiny control set makes $\mu_0$ noisy, polluting
  every prediction.

---

## X-Learner (crossed learner)

**Start with the problem.** Real campaigns are **imbalanced** — e.g. 99% treated, 1% held out. Two
failures bite:

- The T-learner's tiny control model $\mu_0$ is data-starved and noisy.
- Worse, computing uplift as $\mu_1 - \mu_0$ is **"weighing the ship to find the captain":** if
  conversion is 90% baseline + 10% treatment, both models pour their capacity into the 90% baseline,
  and you subtract two large numbers to find a small one — a 1% error in either swamps the effect.

**The idea.** Two moves fix this:

1. **Borrow the big model** to fill each user's missing counterfactual (the large treated group makes
   $\mu_1$ accurate).
2. **Target the gap directly** so the final learner spends its capacity on the *effect*, not the
   baseline.

**Derive it.** Write a treated user's outcome as baseline + effect: $Y(1) = Y(0) + \tau(x)$, so
$\tau(x) = Y(1) - Y(0)$. We know one term and impute the other with the *other group's* model, giving
an **imputed per-user effect**:

- Treated user (know $Y$, impute the no-treatment world): $\;D_1 = Y - \mu_0(x)$
- Control user (know $Y$, impute the treated world): $\;D_0 = \mu_1(x) - Y$

Because $Y$ already *contains* the baseline, subtracting the predicted baseline **strips it away** —
$D$ is (noisy) uplift only. Now fit **new models whose target is the gap**:

$$\tau_1(x) \approx D_1 \;\;(\text{treated}), \qquad \tau_0(x) \approx D_0 \;\;(\text{control})$$

**Combine by trust.** Weight each side by the propensity score $e(x) = P(T=1\mid X=x)$ — lean on
whichever side has more data:

$$\hat{\tau}(x) = e(x)\,\tau_0(x) + \big(1 - e(x)\big)\,\tau_1(x)$$

- **Imbalanced (99% treated):** $e(x) \approx 0.99$ → leans on $\tau_0$. Why is that the *good* one?
  $\tau_0$ trains on labels $D_0 = \mu_1(x) - Y$, and $\mu_1$ came from the *huge* treated group — so
  those labels are high quality despite few control users. The slider routes weight to the pathway
  whose labels were made by the *accurate* base model.
- **Balanced (50/50):** $e(x) = 0.5$ → both sides equally, no data wasted.

**Why keep $D_1$ if the weight goes to $D_0$?**

- **Feature diversity** — $\tau_0$ has clean labels but trains on few control users (overfits, narrow
  feature space); $\tau_1$ has noisier labels but trains on the whole treated population
  (generalizes). The blend is an ensemble of "clean labels / narrow features" + "noisy labels / broad
  features."
- **Generality** — at 50/50 the propensity term gives $D_1$ equal weight automatically; hardcoding
  "use $D_0$ only" would throw away half of a balanced experiment.

> **Key idea:** the X-learner lets the *large* group's intelligence flow into the *small* group's
> estimates via cross-imputation, targets the effect directly (not the baseline), then re-weights by
> group size so a data-starved control set never corrupts the scores.

---

## Choosing among S / T / X

| Learner | Structure | Best when | Main weakness |
|---------|-----------|-----------|---------------|
| **S** | one model, $T$ as feature | few features, want simplicity | can regularize $T$ away → zero uplift |
| **T** | two independent models | balanced groups (≈ 50/50) | high variance; fragile under imbalance |
| **X** | T-stage + cross-imputed re-fit | **imbalanced** groups (small control) | most moving parts; needs a propensity model |

> **Key idea:** start with a **T-learner** when groups are balanced and you want something intuitive;
> upgrade to an **X-learner** when the control group is small or T/S scores look noisy. **EconML**
> (Microsoft) and **CausalML** (Uber) ship all three.
