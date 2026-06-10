# Foundations of Causal Inference

Causal modelling is the leap from *"what is the relationship between $X$ and $Y$?"* (correlation) to
*"if I **change** $X$, how will $Y$ respond?"* (causation). Most real questions are secretly causal —
*Did the notification cause the purchase? Did the drug cause recovery?* — yet the data we collect
usually only speaks correlation. This file builds the intuition for why that gap is hard to close,
before any estimator appears.

---

## Correlation is not causation

Two variables can move together for three different reasons:

1. **$X$ causes $Y$** — the relationship we actually want.
2. **$Y$ causes $X$** — reverse causation.
3. **A third thing $Z$ causes both** — a *confounder*, correlation with no direct link.

Correlation alone can't tell these apart, and acting on the wrong one is expensive: ice-cream sales
correlate with drowning deaths, but banning ice cream saves no one — hot weather (a confounder) drives
both. The field exists to separate case 1 from cases 2 and 3.

> **Key idea:** correlation measures how variables co-vary *in the data you happened to collect*.
> Causation asks what happens *under an intervention you haven't made* — a harder question.

---

## The fundamental problem of causal inference

The effect of a treatment on a *single* unit is a comparison that can never be observed:

> *What did this user do after receiving the notification* — versus — *what would the **same** user
> have done at the **same** moment if they had **not** received it?*

That second world — the **counterfactual** — never happens. You send the message or you don't; you
only ever watch one of the two timelines. This is the **fundamental problem of causal inference**: an
individual effect is the difference between two outcomes, and you can only measure one. The other is
missing data, permanently.

Hence causal inference is a **missing-data problem**. Everything that follows — randomized
experiments, matching, meta-learners — is a strategy for *estimating the outcome we never saw*, using
comparable units as stand-ins for the missing timeline.

> **Key idea:** you can never measure an *individual* causal effect. You can only recover *average*
> effects over groups, by borrowing the missing timeline from comparable units.

---

## Confounding: why naive comparisons lie

The obvious shortcut is to use *other people* as the stand-in: compare those who got the treatment to
those who didn't. This works only if the groups were alike to begin with — they usually aren't.

Say a team sends a WhatsApp message only to users inactive for 30+ days, then compares conversion:

- **Treated** (messaged) = already-disengaged users → low conversion.
- **Untreated** (not messaged) = already-active users → high conversion.

Subtract them and you'd "conclude" the message *hurts* conversion — but the groups differed *before*
any message went out. *Recency* drove both **who got messaged** and **how likely they were to
convert**: a **confounder** opening a fake path between treatment and outcome.

```
                 recency (confounder)
                 /                  \
       (decides who      (also drives
        is messaged)      conversion)
              ↓                  ↓
        treatment  ─── ? ───►  outcome
```

Causal inference must **close this backdoor path** through `recency` — by experiment, matching, or
modelling — leaving only the real causal path from treatment to outcome.

> **Key idea:** a treated-vs-untreated difference is causal only if the groups were *comparable to
> begin with*. Removing confounding to make them comparable is the whole game.

---

## What it means to "control for" something

"Controlling for $X$" means comparing treated and untreated units with the **same value of $X$**, so
$X$ can no longer explain the outcome difference. Comparing only users with the *same recency*
neutralizes that confounder — within each bucket, who got the message looks closer to random.

Two cautions the rest of these notes return to:

- **The right variables remove bias; the wrong ones *add* it.** Conditioning on a variable
  *downstream* of both treatment and outcome (a *collider*) manufactures a correlation that wasn't
  there. More variables is not automatically better.
- **You can only control for what you measured.** Anything unobserved that drives both treatment and
  outcome leaves a residual bias no model can fix — only a randomized experiment or a clever design
  (instrumental variables, discontinuities) can.

---

These three ideas — *correlation ≠ causation*, *the missing counterfactual*, and *confounding* — are
the foundation. The next file makes the target precise: once we accept we can only get average
effects, **which** average do we want? That is the question of **estimands**.
