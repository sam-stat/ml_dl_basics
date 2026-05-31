To understand exactly why Double Machine Learning (DML) works mathematically, we have to look at how errors propagate in high-dimensional estimation. The core of DML relies on a property called Neyman Orthogonality and the mathematics of convergence rates.

Here is the rigorous breakdown of the mechanics.

1. The Formal Setup: The Partially Linear Model
Assume we have N independent observations. We define our data as W=(Y,T,X), where Y is the outcome, T is the treatment, and X is a high-dimensional vector of confounders.

The standard partially linear model is:

Y=θ 
0
​
 T+g 
0
​
 (X)+ϵ
T=m 
0
​
 (X)+η
Where:

θ 
0
​
  is the true causal parameter we want to estimate.

g 
0
​
 (X) and m 
0
​
 (X) are unknown "nuisance functions" (complex, non-linear relationships we must estimate using ML).

E[ϵ∣X,T]=0 and E[η∣X]=0 (the errors are centered).

2. The Mathematical Failure of Naive ML
If we try to estimate θ and g simultaneously (for example, by throwing everything into a neural network and extracting the coefficient for T), we construct an estimator  
θ
^
 .

By the central limit theorem, we want our estimator's error to converge to a normal distribution at a rate of 1/ 
N

​
 . The error decomposition of a naive ML estimator looks like this:

N

​
 ( 
θ
^
 −θ 
0
​
 )= 
N

​
 
1
​
  
i=1
∑
N
​
 ψ 
i
​
 +Bias Term
In a standard approach, the Bias Term scales with  
N

​
 ∣∣ 
g
^
​
 −g 
0
​
 ∣∣.
Because X is high-dimensional, standard machine learning models suffer from the curse of dimensionality. They generally converge at a rate slower than O(N 
−1/2
 ), typically around O(N 
−1/4
 ).

If ∣∣ 
g
^
​
 −g 
0
​
 ∣∣ shrinks at O(N 
−1/4
 ), then the bias term becomes  
N

​
 ×O(N 
−1/4
 )=O(N 
1/4
 ). As N→∞, the bias explodes. The estimator fails completely.

3. The Mathematical Cure: Robinson's Transformation
To fix this, DML applies Robinson's transformation to eliminate g 
0
​
 (X) algebraically.

First, take the conditional expectation of the outcome equation with respect to X:

E[Y∣X]=θ 
0
​
 E[T∣X]+E[g 
0
​
 (X)∣X]+E[ϵ∣X]
Since E[g 
0
​
 (X)∣X]=g 
0
​
 (X) and E[ϵ∣X]=0, this simplifies to:

E[Y∣X]=θ 
0
​
 E[T∣X]+g 
0
​
 (X)
Now, subtract this conditional expectation from the original main equation:

Y−E[Y∣X]=θ 
0
​
 (T−E[T∣X])+ϵ
Let's define our conditional expectations as our two nuisance functions: ℓ 
0
​
 (X)=E[Y∣X] and m 
0
​
 (X)=E[T∣X]. The equation becomes:

Y−ℓ 
0
​
 (X)=θ 
0
​
 (T−m 
0
​
 (X))+ϵ
Let  
Y
~
 =Y−ℓ 
0
​
 (X) and  
T
~
 =T−m 
0
​
 (X). We are left with a simple linear equation:

Y
~
 =θ 
0
​
  
T
~
 +ϵ
4. Neyman Orthogonality
We estimate θ by finding the root of the empirical score function (the derivative of the loss). The score function for our new residualized equation is:

ψ(W;θ,ℓ,m)=((Y−ℓ(X))−θ(T−m(X)))(T−m(X))
This specific score function possesses Neyman Orthogonality. Mathematically, an estimator is Neyman orthogonal if the Gateaux derivative of the expected score with respect to the nuisance parameters (ℓ,m), evaluated at the true parameters, is exactly zero:

∂ 
η
​
 E[ψ(W;θ 
0
​
 ,η 
0
​
 )]=0
Because the first derivative is zero, the estimator is completely insensitive to first-order errors in estimating ℓ(X) and m(X). The first-order bias vanishes.

5. The Convergence Rate Trick (The Magic of N 
−1/4
 )
Because the first-order bias is mathematically eliminated by the orthogonal score, the remaining bias relies only on second-order errors.

Specifically, the new Bias Term is bounded by the product of the estimation errors of the two ML models:

Bias Term≈ 
N

​
 ∣∣ 
ℓ
^
 −ℓ 
0
​
 ∣∣×∣∣ 
m
^
 −m 
0
​
 ∣∣
This is the mathematical masterstroke of DML. If both ML models (the outcome model and the treatment model) converge at the relatively slow, high-dimensional rate of O(N 
−1/4
 ), their product converges at:

O(N 
−1/4
 )×O(N 
−1/4
 )=O(N 
−1/2
 )
When we plug this into our bias equation:

N

​
 ×O(N 
−1/2
 )=O(1)
The  
N

​
  and 1/ 
N

​
  cancel out. The bias is bounded and asymptotically approaches zero. Even though your ML models are fundamentally biased by regularization, their combined product error shrinks fast enough to allow  
θ
^
  to converge at the optimal 1/ 
N

​
  rate, achieving asymptotic normality.

6. Sample Splitting (Cross-Fitting)
There is one final mathematical requirement. The bound O(∣∣ 
ℓ
^
 −ℓ 
0
​
 ∣∣×∣∣ 
m
^
 −m 
0
​
 ∣∣) only holds if the errors of the nuisance models are strictly independent of the data used to estimate θ. If you evaluate the models on the same data used to train them, overfitting creates a correlation that breaks the O(N 
−1/2
 ) convergence.

To satisfy the independence requirement mathematically, DML uses cross-fitting:

Split the data into K folds.

Train  
ℓ
^
  and  
m
^
  on K−1 folds.

Calculate the residuals  
Y
~
  and  
T
~
  on the 1 held-out fold.

Rotate and average the resulting  
θ
^
  estimates.

This ensures all functional approximations are perfectly decoupled from the structural estimation, preserving the orthogonal math.