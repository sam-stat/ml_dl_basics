# Double Machine Learning (DML)

When the path from $X$ to $Y$ is **nonlinear and high-dimensional**, the propensity-score and
meta-learner toolkits both lean on a model that may quietly carry residual confounding through. DML
attacks this differently: use any flexible ML model to **predict away** the influence of $X$ on
both the outcome *and* the treatment, then estimate $\theta$ from the **leftover variation only**.
The framing is Robinson's (1988) partial-out trick, made rigorous for ML nuisance estimators by
Chernozhukov et al. (2018).

Assumes [ignorability](02_estimands.md) given $X$ — like every covariate-adjustment method. What it
doesn't assume is that $X \to Y$ or $X \to T$ is linear; that flexibility is exactly why it works
where OLS partialling-out and logistic-regression [PSM](07_propensity_methods.md) would not.

![DML residualization](diagrams/dml_residualization.svg)

---

## The setup: partially linear model

$$Y = \theta_0\,T + g_0(X) + \varepsilon, \qquad T = m_0(X) + \eta$$

with $E[\varepsilon \mid X, T] = 0$ and $E[\eta \mid X] = 0$.

- $\theta_0$ — the **causal parameter** of interest (a scalar effect of $T$ on $Y$).
- $g_0(X) = E[Y(0) \mid X]$ — how $X$ moves the **baseline outcome**.
- $m_0(X) = E[T \mid X] = e(X)$ — how $X$ moves the **propensity to be treated**.

$g_0$ and $m_0$ are *nuisance* functions: needed to estimate, but not the answer. Their job is to
soak up the parts of $Y$ and $T$ that $X$ already explains, so the leftover slice of $T$ can claim
the leftover slice of $Y$.

---

## The algorithm

Three models, each touching the data through a different door — and each fit on data **disjoint
from the data where it is used**.

- **① Outcome model $\hat\ell(X)$.** Any ML regressor with target $Y$ and features $X$ *only* (no $T$),
  trained to predict $E[Y \mid X]$. Gradient boosting, random forest, ridge, MLP — anything expressive
  enough for the baseline.
  - We regress on $Y$ **as-is**, including the part driven by $T$. We are *not* trying to recover
    $g_0(X)$ in isolation; we are recovering $\ell_0(X) := E[Y \mid X]$. The next section explains
    why this is the right object.

- **② Treatment model $\hat m(X)$.** A regressor (classifier when $T \in \{0,1\}$) with target $T$
  and features $X$, trained to predict $E[T \mid X]$.
  - This is the same *estimand* as the propensity score $e(x)$, but DML uses it as a
    **residualizer** — to strip $X$'s pull out of $T$ — not as a matching key or a weight.

- **③ Cross-fitting — the data plumbing.** Both nuisance models are fit by **K-fold cross-fitting**
  ($K = 5$ or $10$):
  - split the data into $K$ folds;
  - for each fold $k$, train $\hat\ell$ and $\hat m$ on the *other* $K - 1$ folds and predict on
    fold $k$ (out-of-fold);
  - stack so every row has $\hat\ell(x_i)$ and $\hat m(x_i)$ from a model that **never trained on
    $i$**.
  - No nuisance prediction is ever evaluated on a row it saw at training time — this is what keeps
    the orthogonality argument honest (see *Why error stays controlled*).

- **④ Residualize.** Build the leftover variation $X$ couldn't predict:

  $$\tilde Y_i = Y_i - \hat\ell(X_i), \qquad \tilde T_i = T_i - \hat m(X_i)$$

  - $\tilde Y$ = the part of the outcome $X$ couldn't explain.
  - $\tilde T$ = the part of the treatment $X$ couldn't explain.
  - These are the **only** variation $\theta_0$ can claim credit for.

- **⑤ Final stage — regress $\tilde Y$ on $\tilde T$.** A one-variable OLS:

  $$\hat\theta = \frac{\sum_i \tilde T_i \tilde Y_i}{\sum_i \tilde T_i^{\,2}}$$

  - Equivalently, the root of the **DML score**:

  $$\psi(W;\,\theta, \ell, m) = \big(Y - \ell(X) - \theta\,(T - m(X))\big)\big(T - m(X)\big), \qquad \frac{1}{N}\sum_i \psi(W_i;\,\hat\theta,\, \hat\ell, \hat m) = 0$$

  - For the partially linear model this is exactly residual-on-residual OLS. For binary-treatment
    ATE, IV-DML, R-learner CATE, etc., the *score* changes shape — but the framework (cross-fitted
    nuisance + orthogonal score) is identical.

> **Key idea:** two ML models eat the confounding ($\hat\ell$ from the outcome side, $\hat m$ from
> the treatment side, both cross-fitted), and a one-variable OLS on the residuals reads off $\theta$.

---

## Why it works: Robinson's trick

The natural objection: *if I regress $Y$ on $X$ to get $\hat\ell$, but $Y$ already contains
$\theta\,T$, won't $\hat\ell$ swallow part of the treatment effect and bias $\hat\theta$ toward zero?*
The algebra says no — and it starts from the model for $Y$ itself.

- **Start from the outcome model.** The partially linear model says

  $$Y = \theta_0\,T + g_0(X) + \varepsilon$$

- **Take the conditional expectation given $X$.** Average both sides over $X$; since
  $E[g_0(X) \mid X] = g_0(X)$ and $E[\varepsilon \mid X] = 0$:

  $$E[Y \mid X] = \theta_0\,E[T \mid X] + g_0(X) = \theta_0\,m_0(X) + g_0(X)$$

  - So what step ① actually estimates is $\ell_0(X) = \theta_0\,m_0(X) + g_0(X)$ — the slice of $Y$
    that **$X$ alone** can predict, *including* the treatment-mediated piece $\theta_0\,m_0$.

- **Take the difference** $Y - \ell_0(X)$ — subtract the second equation from the first, line by line:

  $$Y - \ell_0(X) \;=\; \underbrace{\theta_0\,T + g_0(X) + \varepsilon}_{Y} \;-\; \underbrace{\theta_0\,m_0(X) - g_0(X)}_{\ell_0(X)} \;=\; \theta_0\,\big(T - m_0(X)\big) + \varepsilon$$

  - The $g_0(X)$ **cancels** — the nuisance baseline is gone without ever isolating it.
  - The $\theta_0\,m_0(X)$ left behind is exactly $\theta_0$ times the **treatment residual**
    $T - m_0(X)$ from step ②.

- **Read off the result.** The treatment signal was not lost — it was **transferred from $Y$ onto
  the residual $\tilde T = T - m(X)$**, where the step-⑤ regression recovers it cleanly as
  $\hat\theta$.

- **The punchline on $\hat g$.** We never estimate $g_0$ in isolation, and we never need to. We
  estimate $\ell_0 = E[Y \mid X]$ — regressing over $Y$ *as a whole* — and that single object
  absorbs both the unwanted baseline $g_0$ **and** the $\theta_0 m_0$ piece along the confounded
  path, leaving a clean linear equation in $\theta_0$.

> **Key idea:** the outcome model targets $E[Y \mid X]$, not $g(X)$. The $\theta\,m(X)$ component
> bundled inside $\ell$ is a feature, not a bug — it's exactly what makes the residual equation
> reduce to $\theta_0\,\tilde T + \varepsilon$.

---

## The score function, and why it debiases

The ML models $\hat\ell$ and $\hat m$ are never exact. The worry is that their errors leak straight
into $\hat\theta$. The machinery that stops the leak lives entirely in the **score function** — so
this section builds it from the ground up: what a score is, how it produces $\hat\theta$, and what we
must demand of it so that imperfect nuisances don't matter.

### What a score function is

A **score** (or *moment function*) $\psi$ is a function of the data $W = (Y, T, X)$, the target
$\theta$, and the nuisances $\eta = (\ell, m)$, built so that its **expectation is zero exactly at the
truth**:

$$E\big[\psi(W;\,\theta, \eta_0)\big] = 0 \quad\Longleftrightarrow\quad \theta = \theta_0.$$

You never minimise a loss directly; you find the $\hat\theta$ that drives the *sample* average of the
score to zero — $\frac{1}{N}\sum_i \psi(W_i;\,\hat\theta, \hat\eta) = 0$. This is all of M-estimation,
including OLS.

- **Linear-regression anchor.** Ordinary least squares *is* a score estimator. Its score is
  "residual $\times$ regressor", $\psi_{\text{OLS}} = X(Y - X^\top\beta)$, and the condition
  $E[X(Y - X^\top\beta_0)] = 0$ just says *the residual must be uncorrelated with the regressor*.
  Setting the sample version to zero gives the normal equations. Keep this picture — DML reuses the
  exact same shape on residualized variables.

### Our score, and how it gives $\hat\theta$

For the partially linear model the DML score is "residual $\times$ residual" (step ⑤):

$$\psi(W;\,\theta, \ell, m) = \big(\underbrace{Y - \ell(X)}_{\tilde Y} - \theta\,\underbrace{(T - m(X))}_{\tilde T}\big)\,\underbrace{(T - m(X))}_{\tilde T} = (\tilde Y - \theta\,\tilde T)\,\tilde T$$

It is the OLS condition again — *the residual of the residual-regression must be uncorrelated with
$\tilde T$* — just with $X$ partialled out of both variables first. Solving $E[\psi] = 0$:

$$E\big[\tilde T(\tilde Y - \theta\,\tilde T)\big] = 0 \;\Longrightarrow\; \hat\theta = \frac{E[\tilde T\,\tilde Y]}{E[\tilde T^{\,2}]}.$$

So the **score defines the estimator**. Everything now rides on one question: when we plug in the
*estimated* $\hat\eta$ rather than the true $\eta_0$, how far does $\hat\theta$ move?

### What this score buys you — the debiasing

The whole reason we use *this* residual-on-residual score, and not a plainer one, is what it does to
the nuisance error. In three lines:

- **Why have it.** This score is shaped so that an error in $\hat\ell$ or $\hat m$ enters $\hat\theta$
  only as a *product* of the two errors — never on its own. A product of two small numbers is
  negligible ($10\% \times 10\% = 1\%$), so even mediocre ML models leave $\hat\theta$ accurate.
- **What it achieves.** A small wobble in either nuisance barely moves $\hat\theta$: the
  error-term that *would* have dominated is removed, and what survives is small enough to vanish as
  fast as ordinary sampling noise. **That removal is the debiasing** — it is built into the score, not
  subtracted by hand afterwards.
- **If we skip it.** Use a plainer score (one ML model, or residualize only one side) and the nuisance
  error enters $\hat\theta$ *directly*. The ML model's regularization shrinkage then leaks straight in,
  pulls $\hat\theta$ toward zero, and no amount of data fixes it.

### Cross-fitting protects this

That guarantee assumes $\hat\ell, \hat m$ never saw the row they score. Train and predict on the same
row and the model partly *memorizes* $Y_i$, quietly re-correlating the residuals — a second, distinct
leak (overfitting) the score's shape doesn't cover. Out-of-fold prediction (step ③) closes it.

> **Key idea:** debiasing *is* the score choice. The residual-on-residual form makes nuisance error
> hit $\hat\theta$ as a *product*, not a sum, so two imperfect ML models still yield a valid causal
> estimate; cross-fitting keeps that guarantee honest.

---

## DML for CATE: the R-learner

Drop the constant-effect assumption and let the effect vary with $x$. The partially linear model
becomes

$$Y = \tau_0(X)\,T + g_0(X) + \varepsilon,$$

and the goal is now the **CATE function** $\tau_0(x)$, not one scalar $\theta_0$. The identical
partial-out algebra (take $E[\cdot \mid X]$, subtract) survives untouched, because $g_0(X)$ still
cancels:

$$\tilde Y = \tau_0(X)\,\tilde T + \varepsilon, \qquad \tilde Y = Y - \ell_0(X), \;\; \tilde T = T - m_0(X).$$

- **The CATE score is conditional on $X$.** Since $\tau$ is now a function, the moment must hold at
  each $x$, not just on average:

  $$\psi(W;\,\tau, \ell, m) = \big(\tilde Y - \tau(X)\,\tilde T\big)\,\tilde T, \qquad E\big[\psi \mid X\big] = 0.$$

- **Solving it is a weighted regression — the R-loss (Nie–Wager).** Integrating the conditional
  moment is equivalent to minimising

  $$\hat\tau = \arg\min_{\tau}\; E\Big[\big(\tilde Y - \tau(X)\,\tilde T\big)^2\Big],$$

  a one-line *weighted* regression: pseudo-outcome $\tilde Y / \tilde T$, weights $\tilde T^2$,
  fittable with any ML learner. This is the **R-learner** — the CATE sibling of scalar DML, and a
  cousin of the [meta-learners](05_meta_learners.md), differing precisely in that it regresses on
  *residualized* treatment.

- **The same debiasing carries over to $\hat\tau$.** Because the R-loss is built from the *two*
  residuals, a small error in $\hat\ell$ or $\hat m$ does not feed into $\hat\tau(x)$ directly — the
  dominant error-term is removed here exactly as it was for the scalar $\hat\theta$. So the R-learner
  inherits DML's guarantee: imperfect, cross-fitted nuisance models still yield a **debiased** CATE
  estimate. Nothing about going from one number $\theta_0$ to a function $\tau_0(x)$ costs you that
  protection.

> **Key idea:** CATE is the same machine with the moment made *conditional on $X$* (equivalently,
> minimise the $\tilde T^2$-weighted R-loss). It debiases for the same reason scalar DML does — the
> residual-on-residual form keeps nuisance error from leaking into $\hat\tau$.
