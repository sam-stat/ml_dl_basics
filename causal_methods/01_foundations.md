# Foundations of Causal Inference

Causal modelling is the leap from *"what is the relationship between $X$ and $Y$?"* (correlation)
to *"if I **change** $X$, how will $Y$ respond?"* (causation). Almost every interesting business or
scientific question is secretly causal — *Did the notification cause the purchase? Did the drug
cause recovery?* — yet the data we have usually only speaks the language of correlation. This file
builds the intuition for why that gap exists and why it is hard to close, before any estimator or
formula appears.

---

## Correlation is not causation

Two variables can move together for three quite different reasons:

1. **$X$ causes $Y$** — the relationship we actually want.
2. **$Y$ causes $X$** — reverse causation.
3. **A third thing $Z$ causes both** — a *confounder* creating a correlation with no direct link.

A correlation alone cannot tell these apart, and acting on the wrong one is expensive. Ice-cream
sales correlate with drowning deaths — but banning ice cream saves no one, because hot weather (a
confounder) drives both. The entire field exists to separate case 1 from cases 2 and 3.

> **Key idea:** correlation measures *how two variables co-vary in the data you happened to
> collect*. Causation asks *what would happen under an intervention you have not yet made* — a
> fundamentally different and harder question.

---

## The fundamental problem of causal inference

The cause-and-effect of a treatment on a *single* unit is defined by a comparison that can never
be observed:

> *What did this user do after receiving the notification* — versus — *what would this **same**
> user have done at the **same** moment if they had **not** received it?*

That second world — the **counterfactual** — never happens. You send the message or you don't; you
only ever get to watch one of the two timelines. This is the **fundamental problem of causal
inference**: for any individual, the effect of a treatment is the difference between two outcomes,
and you can only ever measure one of them. The other is missing data, permanently.

This is why causal inference is often described as a **missing-data problem**. Everything that
follows — randomized experiments, matching, meta-learners — is ultimately a strategy for
*estimating the outcome we never got to see*, using other units as stand-ins for the missing
timeline.

> **Key idea:** you cannot measure an individual causal effect directly — ever. You can only
> recover *average* effects over groups, by borrowing the missing timeline from comparable units.

---

## Confounding: why naive comparisons lie

Because the counterfactual is missing, the obvious shortcut is to use *other people* as the
stand-in: compare those who got the treatment to those who didn't. This works only if the two
groups are otherwise alike. They usually aren't.

Suppose a marketing team sends a WhatsApp message only to users who have been inactive for 30+ days.
Now compare conversion rates:

- **Treated** (messaged) = the already-disengaged users → low conversion.
- **Untreated** (not messaged) = the already-active users → high conversion.

Subtract them and you'd "conclude" the message *hurts* conversion. But the message did nothing of
the sort — the groups were different *before* any message went out. The variable *recency* drove
both **who got the message** and **how likely they were to convert**. That is a **confounder**, and
it has opened a fake path between treatment and outcome.

```
                 recency (confounder)
                 /                  \
       (decides who      (also drives
        is messaged)      conversion)
              ↓                  ↓
        treatment  ─── ? ───►  outcome
```

The job of causal inference is to **close the backdoor path** through `recency` — by experiment, by
matching, or by modelling — so that the only remaining path from treatment to outcome is the real
causal one.

> **Key idea:** a difference in outcomes between treated and untreated groups is only causal if the
> groups were *comparable to begin with*. Making them comparable — removing confounding — is the
> whole game.

---

## What it means to "control for" something

"Controlling for $X$" means comparing treated and untreated units that share the **same value of
$X$**, so $X$ can no longer explain the difference in outcomes. In the example above, comparing only
users with the *same recency* neutralizes that confounder — within each recency bucket, who got the
message looks much closer to random.

Two cautions that the rest of these notes return to:

- **Controlling for the right variables removes bias; controlling for the wrong ones *adds* it.**
  Conditioning on a variable that sits *downstream* of both treatment and outcome (a *collider*) can
  manufacture a correlation that wasn't there. More variables is not automatically better.
- **You can only control for what you measured.** Anything unobserved that drives both treatment and
  outcome leaves a residual bias that no amount of modelling can fix — only a randomized experiment
  or a clever design (instrumental variables, discontinuities) can.

---

These three ideas — *correlation ≠ causation*, *the missing counterfactual*, and *confounding* —
are the foundation. The next file makes the target precise: once we accept we can only get average
effects, **which** average do we actually want? That is the question of **estimands**.
