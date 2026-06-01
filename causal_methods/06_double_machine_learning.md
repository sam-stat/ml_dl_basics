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

## Why the error stays controlled: Neyman orthogonality

The ML models $\hat\ell$ and $\hat m$ are never exact. The worry is that their errors leak straight
into $\hat\theta$. **Neyman orthogonality** is the property that stops this — and note it is *not* an
assumption about the data (like ignorability); it's a property we **engineer into the score** by
choosing the residual-on-residual form.

- **Orthogonality = the score is flat in the nuisance direction.** Think of the *expected* score
  $E[\psi]$ as a function of the nuisances $\eta = (\ell, m)$, with $\theta$ held at its true value
  $\theta_0$. The condition is that its **derivative with respect to $\eta$, evaluated at the true
  nuisances $\eta_0$, vanishes**:

  $$\partial_\eta\, E\big[\psi(W;\,\theta_0,\, \eta)\big]\big|_{\eta = \eta_0} = 0, \qquad \eta = (\ell, m)$$

  - This is a derivative *in the function direction*: it asks how $E[\psi]$ responds when you perturb
    the whole function $\ell$ (or $m$) slightly away from the truth $\ell_0$ (or $m_0$) — a
    directional (Gâteaux) derivative, not an ordinary $\partial/\partial x$.
  - Its being zero means $E[\psi]$ sits at a **flat spot** in the nuisance direction at $\eta_0$. So
    nudging $\hat\ell$ or $\hat m$ slightly off the truth barely moves $\theta$'s estimating equation —
    a single nuisance error can't tilt $\hat\theta$.

- **Why the first derivative is the thing that matters — Taylor's view.** Let $\delta = \hat\eta -
  \eta_0$ be the nuisance error (small, since the ML models are decent). The bias in $\hat\theta$ is a
  function of $\delta$, and a Taylor expansion around the truth $\delta = 0$ reads:

  $$\text{bias}(\delta) \;=\; \underbrace{\text{bias}(0)}_{=\,0} \;+\; \underbrace{b'\,\delta}_{\text{first order}} \;+\; \underbrace{\tfrac{1}{2}\,b''\,\delta^2}_{\text{second order}} \;+\; \cdots$$

  - $\text{bias}(0) = 0$: plug in the *true* nuisances and the score gives the right answer.
  - The **first-order term $b'\delta$** is *linear* in the error. Because $\delta$ is small,
    $\delta \gg \delta^2$, so this term **dominates** — whatever rate $b'\delta$ shrinks at is the rate
    the whole bias shrinks at. Its coefficient $b'$ is exactly the derivative $\partial_\eta E[\psi]$
    from above.
  - **That is why we care about the first derivative:** it is the coefficient of the dominant term.
    Orthogonality *sets $b' = 0$*, deleting the entire linear term and promoting the much smaller
    second-order term to be the leader.

- **What's left is only a product.** With $b' = 0$, the leading survivor is the **second-order term** —
  and for the DML score it takes the form of a *product* of the two separate errors (not a square of
  one):

  $$\big|\,\hat\theta - \theta_0\,\big| \;\lesssim\; \big\|\hat\ell - \ell_0\big\| \cdot \big\|\hat m - m_0\big\|$$

  A product of two small numbers is tiny: $10\% \times 10\% = 1\%$. Both models can be mediocre and
  $\hat\theta$ stays accurate.

- **Cross-fitting protects the product.** That bound only holds if $\hat\ell$ and $\hat m$ didn't
  train on the row they're scoring; otherwise overfitting reintroduces a correlation. Out-of-fold
  prediction (step ③) guarantees it.

> **Key idea:** orthogonality makes nuisance error hit $\hat\theta$ as a *product*, not a sum, so
> two imperfect ML models still yield a valid causal estimate.
