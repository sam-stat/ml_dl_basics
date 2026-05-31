Can you help me understand DML from basics?

DML can mean several things. I’ll assume you mean Double/Debiased Machine Learning, the causal-inference method; I’ll also note the database meaning briefly so you can redirect if needed.

Thought for 12s

DML usually means one of two things:

Double/Debiased Machine Learning in statistics, econometrics, and causal inference.
Data Manipulation Language in SQL, meaning commands like SELECT, INSERT, UPDATE, DELETE.

Assuming you mean Double Machine Learning, here is the basic idea.

1. What problem does DML solve?

Suppose you want to estimate a causal effect:

What is the effect of treatment D on outcome Y,

I mean double machine learnong?
Thought for a couple of seconds

Yes — Double Machine Learning, often written DML, is a method used mainly in causal inference to estimate treatment effects while using flexible machine learning models.

The goal is usually to answer questions like:

What is the effect of a treatment/intervention D on an outcome Y, after adjusting for confounders X?

Examples:

What is the effect of job training on income?

What is the effect of a medicine on recovery?

What is the effect of education on wages?

1. The basic causal problem

Suppose we observe:

Y=outcome
D=treatment
X=control variables / confounders

Example:

Variable	Meaning
Y	Income
D	Whether someone received job training
X	Age, education, prior income, location, etc.

We want to estimate the causal effect of D on Y.

A simple regression might be:

Y=θD+βX+ϵ

Here, θ is the treatment effect.

But in real life, the relationship between X, D, and Y may be complicated and nonlinear. That is where machine learning helps.

2. Why not just use machine learning directly?

You might think:

Let’s use random forests, boosting, neural networks, or Lasso to estimate the treatment effect.

But ordinary machine learning is usually optimized for prediction, not causal inference.

Machine learning may predict Y very well, but it can still produce biased estimates of the treatment effect θ.

The problem is that ML models can overfit, introduce regularization bias, or accidentally mix up the effect of D with the effect of X.

DML fixes this by separating two tasks:

Use machine learning to handle the complicated control variables.
Use a debiased statistical step to estimate the treatment effect.
3. The main idea of DML

DML estimates the treatment effect after removing the influence of confounders X.

Think of it in two steps.

Suppose we want to estimate the effect of D on Y.

First, predict Y using X:

Y
^
=f(X)

Then take the residual:

Y
~
=Y−
Y
^

This is the part of Y not explained by X.

Second, predict D using X:

D
^
=g(X)

Then take the residual:

D
~
=D−
D
^

This is the part of D not explained by X.

Finally, regress:

Y
~
 on 
D
~

The coefficient gives the DML estimate of the treatment effect.

In simple terms:

DML removes the effect of controls from both the outcome and the treatment, then estimates the relationship between the remaining unexplained parts.

4. Why is it called “double”?

It is called double because machine learning is used twice:

To estimate the outcome model:
E[Y∣X]
To estimate the treatment model:
E[D∣X]

These two models are often called nuisance functions.

They are important, but they are not the main target. The main target is the treatment effect θ.

5. Why is it called “debiased”?

Machine learning models can introduce bias because of overfitting or regularization.

DML uses a special correction called orthogonalization or Neyman orthogonality.

The practical idea is:

Small errors in the machine learning predictions should not strongly distort the treatment effect estimate.

So DML is designed to make the final estimate less sensitive to mistakes in the nuisance models.

6. Why cross-fitting is important

DML usually uses cross-fitting.

The data is split into folds.

For example, with two folds:

Train ML models on Fold 1, predict on Fold 2.
Train ML models on Fold 2, predict on Fold 1.
Combine the out-of-sample residuals.
Estimate the treatment effect.

This avoids using the same data both to train the ML model and estimate the final causal effect.

That reduces overfitting bias.

7. A simple example

Suppose we study the effect of job training on income.

Y: income after training
D: whether person received training
X: age, education, previous income, gender, location, work experience

DML does this:

First, use ML to predict income from background variables:

Y∼X

Then calculate:

Y
~
=actual income−predicted income

Second, use ML to predict training participation from background variables:

D∼X

Then calculate:

D
~
=actual training status−predicted training status

Finally:

Y
~
=θ
D
~
+error

The estimate of θ is interpreted as the causal effect, assuming the usual causal assumptions hold.

8. The key assumptions

DML is powerful, but it does not magically prove causality. It still needs causal assumptions.

The main assumptions are:

1. No unobserved confounding

All important confounders must be included in X.

That means, after controlling for X, treatment assignment is as good as random.

2. Overlap

For different values of X, there must be some people who receive treatment and some who do not.

If only one type of person gets treatment, comparison becomes impossible.

3. Correct target parameter

You need to know what effect you want: average treatment effect, partial linear coefficient, treatment effect on treated, etc.

9. Basic DML formula for the partial linear model

A common DML setup is:

Y=θD+g(X)+U
D=m(X)+V

where:

Y is outcome
D is treatment
X are controls
g(X) captures how controls affect outcome
m(X) captures how controls affect treatment
θ is the causal parameter of interest

DML estimates:

Y
~
=Y−
g
^
	​

(X)
D
~
=D−
m
^
(X)

Then:

θ
^
=
∑
i
	​

D
~
i
2
	​

∑
i
	​

D
~
i
	​

Y
~
i
	​

	​


That is just the slope from regressing residualized Y on residualized D.

10. Where DML is useful

DML is useful when:

You have many control variables.
Relationships are nonlinear.
You want causal effects, not just predictions.
You want to combine machine learning with valid statistical inference.
You want standard errors and confidence intervals for treatment effects.

Common fields:

Economics
Public policy
Biostatistics
Marketing
Social science
Epidemiology
Labor studies
11. DML in one sentence

Double Machine Learning uses machine learning to remove the effects of confounders from both the treatment and the outcome, then estimates the causal effect using the remaining variation.

A good learning path is: ordinary regression → confounding → residualization → causal inference assumptions → cross-fitting → orthogonal scores → DML implementation.

I’ll focus on the orthogonality part, because that is where “first-stage error” and “second-order error” become meaningful. The key idea is not that the ML models are perfect, but that their small errors do not enter the treatment-effect estimate linearly.

Thought for 11s

Here “3” refers to Neyman orthogonality, the main debiasing idea in DML.

There are two phrases that sound similar but mean different things:

Phrase	Meaning
First-stage error	Error in estimating nuisance ML functions like (E[Y
First-order error	Error that affects the final 
θ
^
 linearly
Second-order error	Error that affects 
θ
^
 only through products or squares of first-stage errors
1. What is “first-stage error”?

In DML, the first stage estimates nuisance functions:

l
0
	​

(X)=E[Y∣X]
m
0
	​

(X)=E[D∣X]

But we do not know the true functions. We estimate them using ML:

l
^
(X)
m
^
(X)

The first-stage errors are:

δ
l
	​

(X)=
l
^
(X)−l
0
	​

(X)
δ
m
	​

(X)=
m
^
(X)−m
0
	​

(X)

So:

δ
l
	​

(X) = error in predicting the outcome from X
δ
m
	​

(X) = error in predicting the treatment from X

These are called first-stage errors because l and m are estimated before the final treatment effect θ.

They are not called first-stage because they are “first-order.” That is a separate idea.

2. What would go wrong without orthogonality?

Suppose our final treatment-effect estimator depends on 
l
^
 and 
m
^
.

If small errors in 
l
^
 or 
m
^
 directly create bias in 
θ
^
, then we may have something like:

bias in 
θ
^
≈C
1
	​

δ
l
	​

+C
2
	​

δ
m
	​


This is first-order bias.

It means that if your nuisance model has error of size 0.10, the final treatment effect may also have error roughly of size 0.10.

That is bad, because ML nuisance models are never perfect.

3. What does orthogonality do?

DML chooses a score/moment equation such that the first derivative with respect to nuisance errors is zero.

The DML score is:

ψ(W;θ,l,m)=(Y−l(X)−θ(D−m(X)))(D−m(X))

At the true values (θ
0
	​

,l
0
	​

,m
0
	​

), DML wants:

E[ψ(W;θ
0
	​

,l
0
	​

,m
0
	​

)]=0

But we do not use l
0
	​

,m
0
	​

. We use 
l
^
,
m
^
.

Orthogonality says:

∂l
∂
	​

E[ψ(W;θ
0
	​

,l,m
0
	​

)]
	​

l=l
0
	​

	​

=0

and

∂m
∂
	​

E[ψ(W;θ
0
	​

,l
0
	​

,m)]
	​

m=m
0
	​

	​

=0

Meaning:

Small mistakes in 
l
^
 and 
m
^
 do not create first-order bias in 
θ
^
.

So instead of:

bias≈C
1
	​

δ
l
	​

+C
2
	​

δ
m
	​


we get something like:

bias≈C
3
	​

δ
l
	​

δ
m
	​

+C
4
	​

δ
m
2
	​


These are second-order errors.

4. What is a second-order error?

A second-order error is a product or square of smaller errors.

For example, suppose:

δ
l
	​

=0.10
δ
m
	​

=0.10

A first-order error would be around:

0.10

A second-order error could be:

δ
l
	​

δ
m
	​

=0.10×0.10=0.01

or:

δ
m
2
	​

=0.10
2
=0.01

So the error drops from about 10% scale to about 1% scale.

That is the intuition behind debiasing.

5. Taylor expansion intuition

Think of some function F(η), where η represents nuisance functions like l,m.

Suppose the true nuisance value is η
0
	​

, but we use:

η
^
	​

=η
0
	​

+δ

A Taylor expansion gives:

F(η
0
	​

+δ)=F(η
0
	​

)+F
′
(η
0
	​

)δ+
2
1
	​

F
′′
(η
0
	​

)δ
2
+⋯

If:

F
′
(η
0
	​

)

=0

then the leading error is:

F
′
(η
0
	​

)δ

That is first-order error.

But if DML constructs the score so that:

F
′
(η
0
	​

)=0

then the first-order term disappears.

Now the leading error is:

2
1
	​

F
′′
(η
0
	​

)δ
2

That is second-order error.

This is why DML wants the derivative with respect to nuisance functions to be zero.

6. In DML, what exactly becomes second order?

Let:

l
^
(X)=l
0
	​

(X)+δ
l
	​

(X)
m
^
(X)=m
0
	​

(X)+δ
m
	​

(X)

The DML score uses:

(Y−
l
^
(X)−θ(D−
m
^
(X)))(D−
m
^
(X))

Because of orthogonality, the remaining bias from nuisance estimation is not mainly:

δ
l
	​


or:

δ
m
	​


Instead, it is roughly of the form:

δ
l
	​

δ
m
	​


and:

δ
m
2
	​


So the ML errors enter the final estimate only through smaller product/square terms.

That is the “debiased” part.

7. Simple summary

The first-stage errors are:

l
^
(X)−l
0
	​

(X)

and:

m
^
(X)−m
0
	​

(X)

They are the errors made by the ML models.

A first-order error would mean these mistakes directly contaminate 
θ
^
.

A second-order error means the final contamination is only through terms like:

(
l
^
−l
0
	​

)(
m
^
−m
0
	​

)

or:

(
m
^
−m
0
	​

)
2

DML is designed so that the first-order effect disappears. That is what Neyman orthogonality gives you.

where is this score getting used? 
Thought for a few seconds

The score is used in the final stage, after estimating the nuisance functions.

It is not usually used to train the ML models 
l
^
(X) and 
m
^
(X).

Full DML pipeline

Suppose:

l
0
	​

(X)=E[Y∣X]
m
0
	​

(X)=E[D∣X]

DML does this:

Step 1: Estimate nuisance functions

Use ML to estimate:

l
^
(X)≈E[Y∣X]
m
^
(X)≈E[D∣X]

This is the first stage.

For example:

Y∼X
D∼X
Step 2: Construct residuals
Y
~
i
	​

=Y
i
	​

−
l
^
(X
i
	​

)
D
~
i
	​

=D
i
	​

−
m
^
(X
i
	​

)
Step 3: Use the score to estimate θ

The DML score is:

ψ
i
	​

(θ)=(
Y
~
i
	​

−θ
D
~
i
	​

)
D
~
i
	​


or equivalently:

ψ
i
	​

(θ)=(Y
i
	​

−
l
^
(X
i
	​

)−θ(D
i
	​

−
m
^
(X
i
	​

)))(D
i
	​

−
m
^
(X
i
	​

))

Then DML chooses 
θ
^
 so that the average score is zero:

n
1
	​

i=1
∑
n
	​

ψ
i
	​

(
θ
^
)=0

That is where the score is used.

What does setting the score to zero mean?

It means:

n
1
	​

i
∑
	​

(
Y
~
i
	​

−
θ
^
D
~
i
	​

)
D
~
i
	​

=0

This says:

After choosing 
θ
^
, the remaining error 
Y
~
i
	​

−
θ
^
D
~
i
	​

 should be uncorrelated with residualized treatment 
D
~
i
	​

.

This is exactly the OLS condition for regressing residualized Y on residualized D.

Solving it gives:

θ
^
=
∑
i
	​

D
~
i
2
	​

∑
i
	​

D
~
i
	​

Y
~
i
	​

	​


So in the partially linear DML case, the score is used in the final residual-on-residual regression.

Important distinction

The ML models are trained using their own losses:

l
^
:
l
min
	​

i
∑
	​

(Y
i
	​

−l(X
i
	​

))
2
m
^
:
m
min
	​

i
∑
	​

(D
i
	​

−m(X
i
	​

))
2

or logistic loss if D is binary.

But θ is estimated using the orthogonal score:

n
1
	​

i
∑
	​

(Y
i
	​

−
l
^
(X
i
	​

)−θ(D
i
	​

−
m
^
(X
i
	​

)))(D
i
	​

−
m
^
(X
i
	​

))=0

So:

The score is used to estimate θ, not to train the nuisance ML models.

Why not just say “OLS on residuals”?

For this simple partially linear model, you can say that.

But the score formulation is more general.

It is useful because:

It shows exactly where debiasing happens.
It works beyond simple OLS-style DML.
It gives standard errors and confidence intervals.
It lets us check orthogonality mathematically.

In this model:

score equation=OLS first-order condition on residuals

So the score is not an extra mysterious object. It is the mathematical equation behind the final DML estimate.

For DML, this score function is always used. If this code function is always used and the equation is always linear in case of DML, then why is OLS style estimation different from DML, and what do you mean by it was beyond simple OLS? 
Thought for 8s

No. The score function we wrote is not always the DML score. It is the score for one common DML model: the partially linear regression model.

That model is:

Y=θD+h(X)+U

For this case, the DML score is:

ψ(W;θ,l,m)=(Y−l(X)−θ(D−m(X)))(D−m(X))

and the final estimate is equivalent to OLS of:

Y−
l
^
(X)

on:

D−
m
^
(X)

So for this special DML model, yes, the final stage looks like residualized OLS.

But DML as a framework is broader.

1. OLS and DML are different because OLS assumes a simple control structure

Ordinary OLS would estimate:

Y=θD+X
′
β+ϵ

This assumes the effect of controls is linear:

h(X)=X
′
β

DML instead allows:

Y=θD+h(X)+U

where h(X) can be complicated: nonlinear, interactions, high-dimensional, etc.

So OLS assumes:

h(X)=β
1
	​

X
1
	​

+β
2
	​

X
2
	​

+⋯

DML allows:

h(X)=unknown complicated function

estimated by random forest, boosting, neural nets, Lasso, etc.

That is one major difference.

2. OLS partialling-out and DML partialling-out are related

OLS already has a partialling-out idea.

By the Frisch-Waugh-Lovell theorem, the coefficient on D in:

Y∼D+X

is equivalent to:

Regress Y on X, get residuals.
Regress D on X, get residuals.
Regress residualized Y on residualized D.

So OLS also does:

Y−
E
^
linear
	​

[Y∣X]

on:

D−
E
^
linear
	​

[D∣X]

But OLS uses linear projections.

DML uses machine-learning estimates:

l
^
(X)≈E[Y∣X]
m
^
(X)≈E[D∣X]

plus cross-fitting and orthogonal scores to avoid overfitting and regularization bias.

So in the partially linear case:

DML≈ML-based, cross-fitted, orthogonalized version of OLS partialling-out
3. When would DML become the same as OLS?

If you estimate l(X) and m(X) using ordinary linear regression, and you do not use any flexible ML, then DML partialling-out becomes very close to ordinary OLS.

For example:

l
^
(X)=
α
^
Y
	​

+X
′
β
^
	​

Y
	​

m
^
(X)=
α
^
D
	​

+X
′
β
^
	​

D
	​


Then residualizing Y and D, followed by residual-on-residual regression, gives the OLS coefficient on D from:

Y∼D+X

So DML is not “magically different” from OLS in that narrow case.

The difference appears when X is complex and the nuisance functions are estimated flexibly.

4. What does “beyond simple OLS” mean?

It means DML is not only for this model:

Y=θD+h(X)+U

DML is a general method for estimating causal parameters using orthogonal scores.

Different causal questions require different scores.

Example 1: Partially linear regression

Target:

Y=θD+h(X)+U

Score:

(Y−l(X)−θ(D−m(X)))(D−m(X))

Final step: residualized OLS.

This is the case we have been discussing.

Example 2: Average treatment effect with binary treatment

Suppose:

D∈{0,1}

and we want:

θ=E[Y(1)−Y(0)]

Now the score is not residualized OLS. A common orthogonal score is:

ψ(W;θ,η)=μ
1
	​

(X)−μ
0
	​

(X)+
p(X)
D(Y−μ
1
	​

(X))
	​

−
1−p(X)
(1−D)(Y−μ
0
	​

(X))
	​

−θ

where:

μ
1
	​

(X)=E[Y∣D=1,X]
μ
0
	​

(X)=E[Y∣D=0,X]
p(X)=P(D=1∣X)

Then θ is estimated by solving:

n
1
	​

i
∑
	​

ψ(W
i
	​

;θ,
η
^
	​

)=0

which gives:

θ
^
=
n
1
	​

i
∑
	​

[
μ
^
	​

1
	​

(X
i
	​

)−
μ
^
	​

0
	​

(X
i
	​

)+
p
^
	​

(X
i
	​

)
D
i
	​

(Y
i
	​

−
μ
^
	​

1
	​

(X
i
	​

))
	​

−
1−
p
^
	​

(X
i
	​

)
(1−D
i
	​

)(Y
i
	​

−
μ
^
	​

0
	​

(X
i
	​

))
	​

]

This is not OLS on residuals. It is an orthogonal augmented inverse probability weighted estimator.

Example 3: Instrumental variables DML

Suppose D is endogenous and we have an instrument Z.

Then a DML score may look like:

(Y−l(X)−θ(D−m(X)))(Z−r(X))

where:

r(X)=E[Z∣X]

The final stage is closer to IV estimation, not ordinary OLS.

So “beyond simple OLS” means:

The partially linear DML score happens to look like residualized OLS, but other DML scores handle treatment effects, binary treatments, IV settings, heterogeneous effects, and nonlinear causal parameters.

5. Is the DML equation always linear in θ?

No.

In the partially linear model, the score is linear in θ:

(Y−l(X)−θ(D−m(X)))(D−m(X))

So we can solve for 
θ
^
 directly.

But in more general DML problems, the score may be nonlinear in θ, and we may need numerical root-finding:

n
1
	​

i
∑
	​

ψ(W
i
	​

;θ,
η
^
	​

)=0

The general DML idea is not “run OLS.” The general idea is:

Define an orthogonal score for the causal parameter.
Estimate nuisance functions using ML.
Use cross-fitting.
Solve the empirical moment condition:
n
1
	​

i
∑
	​

ψ(W
i
	​

;θ,
η
^
	​

)=0

In the partially linear case, this happens to reduce to residualized OLS.

6. Main connection

OLS says:

Y=θD+X
′
β+ϵ

DML says:

Y=θD+h(X)+U

OLS controls for X using a linear model.

DML controls for X using flexible ML and an orthogonal score.

So:

DML is not different from OLS because the final formula is always different.
DML is different because the nuisance adjustment is flexible, cross-fitted, and orthogonalized.

For the partially linear model, DML is best understood as:

OLS partialling-out generalized with ML and debiasing.