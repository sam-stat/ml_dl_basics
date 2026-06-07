Can you explain me about causal forest from basics for CATE estimation. Explain the algorithm in details with all the maths components

To understand the Conditional Average Treatment Effect (CATE) and how Causal Forests estimate it, we first need to look at the mathematical goal of causal inference and why traditional machine learning falls short.

1. The Goal: CATE Estimation
In standard predictive modeling, the goal is to predict an outcome Y given a set of features X. In causal inference, the goal is to predict the change in an outcome caused by a specific intervention or treatment W.

Imagine you are designing an optimization engine to allocate loan offers.

W (Treatment): Receiving a premium low-interest loan offer (W=1) versus a standard offer (W=0).

Y (Outcome): Whether the user accepts the loan (or alternatively, whether they default).

X (Covariates): User features like income, credit history, and past loan behavior.

You want to know the effect of the premium offer for a specific user profile x to optimize your distribution of High, Medium, and Low risk. This is the CATE, mathematically defined as:

τ(x)=E[Y(1)−Y(0)∣X=x]
Where Y(1) is the potential outcome if the user receives the premium offer, and Y(0) is the potential outcome if they receive the standard offer.

The Fundamental Problem: For any given user, you only ever observe one reality. If a user received the premium offer, you observe Y 
i
​
 (1), but Y 
i
​
 (0) remains a counterfactual. Because the target variable—the individual treatment effect Y 
i
​
 (1)−Y 
i
​
 (0)—is completely unobservable, you cannot use a standard Random Forest to estimate it directly (Wager & Athey, 2015).

2. From Random Forests to Causal Forests
A standard Random Forest averages the predictions of many decision trees to estimate E[Y∣X=x]. A Causal Forest adapts this architecture to estimate E[Y(1)−Y(0)∣X=x]. It achieves this by changing two core components: how the trees split and how the final prediction is calculated.

To do this mathematically, the algorithm relies on two foundational assumptions:

Unconfoundedness: W 
i
​
 ⊥⊥(Y 
i
​
 (1),Y 
i
​
 (0))∣X 
i
​
 . Conditional on the user's features, the treatment assignment is essentially random (all confounders are observed).

Overlap: 0<P(W 
i
​
 =1∣X 
i
​
 =x)<1. Every user profile has a non-zero probability of receiving either treatment.

3. The Mathematical Components of a Causal Forest
Modern Causal Forests (specifically implemented via the Generalized Random Forest framework) utilize a three-part mathematical strategy: Orthogonalization, Honest Splitting, and Local Weighting.

A. Orthogonalization (Robinson's Transformation)
To isolate the causal effect from the baseline characteristics of the users, the algorithm uses a statistical technique called residualization. We define two "nuisance parameters":

μ(X 
i
​
 )=E[Y 
i
​
 ∣X 
i
​
 ]: The expected baseline outcome (e.g., the baseline probability of taking a loan regardless of the offer).

e(X 
i
​
 )=P(W 
i
​
 =1∣X 
i
​
 ): The propensity score (the probability of receiving the premium offer based on features).

First, the algorithm trains standard machine learning models to estimate  
μ
^
​
 (X 
i
​
 ) and  
e
^
 (X 
i
​
 ). It then calculates the residuals:

Y
~
  
i
​
 =Y 
i
​
 − 
μ
^
​
 (X 
i
​
 )
W
~
  
i
​
 =W 
i
​
 − 
e
^
 (X 
i
​
 )
This isolates the unexpected variation.  
Y
~
  
i
​
  represents "did this user accept the loan more or less than we expected?" and  
W
~
  
i
​
  represents "did this user get the premium offer when we didn't expect them to?"

B. Honest Tree Construction and Splitting
A major contribution of the Causal Forest algorithm is the concept of Honest Trees (Wager & Athey, 2015). To prevent overfitting, the training data for each tree is split in half:

Splitting Subsample: Used strictly to determine where the tree splits.

Estimating Subsample: Pushed down the finalized tree structure to calculate the actual treatment effects in the leaf nodes.

The Splitting Criterion: Standard decision trees split to minimize the Mean Squared Error (MSE) of the outcome. A Causal Tree instead splits to maximize the heterogeneity of the treatment effect.
For a given parent node, the algorithm evaluates potential splits into left (L) and right (R) child nodes to maximize:

N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 
Where N is the number of samples in the node, and  
τ
^
  is the estimated Average Treatment Effect (ATE) within that node. Maximizing this equation forces the tree to find boundaries in the feature space X that sharply separate users who respond strongly to the treatment from those who do not.

C. Adaptive Local Weighting
In a standard forest, you drop a new profile x down all trees, find the leaf nodes it lands in, and average the values in those leaves.

Causal Forests use the trees differently. They use the forest to generate an adaptive weighting function, α 
i
​
 (x). This weight represents how frequently a training sample i falls into the exact same leaf node as the target profile x across all B trees in the forest.

α 
i
​
 (x)= 
B
1
​
  
b=1
∑
B
​
  
∣L 
b
​
 (x)∣
I(X 
i
​
 ∈L 
b
​
 (x))
​
 
(Where L 
b
​
 (x) is the leaf in tree b containing x, and ∣L 
b
​
 (x)∣ is the number of training samples in that leaf).

D. The Final CATE Estimation
Finally, to estimate the CATE for the specific profile x, the algorithm solves a locally weighted least squares regression using the residuals from Step A and the weights from Step C:

τ
^
 (x)=arg 
τ
min
​
  
i=1
∑
n
​
 α 
i
​
 (x)( 
Y
~
  
i
​
 −τ 
W
~
  
i
​
 ) 
2
 
By minimizing this objective function, we arrive at the closed-form solution for the CATE:

τ
^
 (x)= 
∑ 
i=1
n
​
 α 
i
​
 (x) 
W
~
  
i
2
​
 
∑ 
i=1
n
​
 α 
i
​
 (x) 
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
4. Summary of the Algorithm Steps
To put it all together into an execution pipeline:

Cross-Fitting: Split the data into folds. Train models to predict the expected outcome  
μ
^
​
 (X 
i
​
 ) and propensity score  
e
^
 (X 
i
​
 ) out-of-fold.

Residualize: Compute the orthogonalized targets  
Y
~
  
i
​
  and  
W
~
  
i
​
 .

Grow the Forest: Construct B causal trees. For each tree:

Subsample the data.

Divide the subsample into a splitting set and an estimating set (Honesty).

Use the splitting set to grow the tree by maximizing treatment effect variance.

Calculate Weights: For a new user profile x, traverse all B trees to determine the similarity weights α 
i
​
 (x) relative to the training data.

Compute CATE: Calculate the final  
τ
^
 (x) via the locally weighted regression equation.

This methodology allows you to move beyond predicting what a user will do, to predicting how they will react to a specific, engineered intervention.

References

Wager, S., & Athey, S. (2015). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. arXiv. https://doi.org/10.48550/arxiv.1510.04342
Cited by: 4956

Not able to understand all the components of it.

Can you first explain how the splitting is done?

It is completely understandable that the mechanics of Causal Forests feel a bit abstract. It is a dense algorithm, and the way it builds its trees is the most crucial difference between traditional machine learning and causal inference.

Let's isolate exactly how a Causal Tree decides to split the data.

1. The Core Difference: Prediction vs. Intervention
Before looking at the math, we need to shift our objective.

Imagine we are managing a lending optimization engine and we want to know how a specific user will respond to a premium loan offer (W=1) versus a standard offer (W=0). We have historical data on users, including their calculated High, Medium, and Low (HML) risk scores.

A Standard Decision Tree splits data to group users who have the same outcome. It asks: "Regardless of the offer, how do I group users so I can accurately predict if they will accept the loan?" It splits to minimize prediction error (Mean Squared Error).

A Causal Tree splits data to group users who have the same reaction to the intervention. It asks: "How do I split this HML risk distribution so that one group is highly influenced by the premium offer, and the other group doesn't care?" It splits to maximize the difference in treatment effects between the two groups.

2. The Math of the Split (Maximizing Heterogeneity)
When the Causal Tree is deciding where to draw a line in the data (e.g., "Should I split users at a Risk Score of 65?"), it evaluates that potential split using a specific objective function.

It wants to find a split that creates a Left child node (L) and a Right child node (R) where the treatment effect ( 
τ
^
 ) is as drastically different as possible.

The algorithm calculates the estimated treatment effect in the proposed left node ( 
τ
^
  
L
​
 ) and the proposed right node ( 
τ
^
  
R
​
 ). The treatment effect is simply the average outcome of the treated users minus the average outcome of the control users in that specific node.

It then evaluates the split using this formula:

Maximize:N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 
(Where N 
L
​
  and N 
R
​
  are the number of users in the left and right nodes).

Why this works:
Squaring the treatment effects penalizes splits that result in homogeneous reactions.

A Bad Split: If both the Left and Right nodes show that the premium offer increases loan acceptance by 3% ( 
τ
^
  
L
​
 =0.03,  
τ
^
  
R
​
 =0.03), the variance between the groups is zero. The split didn't help us personalize anything; everyone reacts the same way.

A Good Split: If the Left node (e.g., Low Risk) shows a 15% increase in acceptance ( 
τ
^
  
L
​
 =0.15), and the Right node (e.g., High Risk) shows a 0% increase ( 
τ
^
  
R
​
 =0.00), the variance is massive. The algorithm heavily favors this split because it perfectly isolated the group that actually responds to the premium offer.

3. "Honesty": Splitting vs. Estimating
There is a massive statistical trap in the splitting process described above: Overfitting the treatment effect.

If you use your dataset to search for the split that creates the absolute largest difference in  
τ
^
 , and then you use that exact same dataset to confidently declare, "The treatment effect in this leaf is 15%," your estimate will be artificially inflated. You essentially just hunted through the noise for the most extreme outcome and graded your own homework.

To solve this, Causal Forests use Honest Splitting. Before a tree is built, the algorithm takes the subsample of data allocated to that tree and chops it in half again:

The Splitting Subsample: This half of the data is pushed down the root node. The algorithm uses it strictly to test different splits and evaluate the N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
  formula. It determines the "geography" of the tree (e.g., "Split at Risk Score = 65").

The Estimating Subsample: Once the tree's branches are finalized, the Splitting Subsample is thrown away. The algorithm takes the second half of the data, drops it down the newly constructed branches, and calculates the actual  
τ
^
  values in the final leaf nodes.

Because the users used to estimate the effect are entirely different from the users used to draw the boundaries, the resulting causal estimate is "honest" and statistically valid.

To really solidify this, let's look at how the mathematics of that objective function play out visually. I've set up an interactive simulation where you can manually test different split points on a simulated risk score feature to see how it impacts the calculated treatment effects and the splitting objective score.



how largest difference in $\hat{\tau}$ is exactly related to maximising $$Maximize: \quad N_L \hat{\tau}_L^2 + N_R \hat{\tau}_R^2$$?

That is a fantastic question. It hits exactly on the mathematical "trick" that makes Causal Forests computationally efficient.

You are asking: How do we get from the intuitive goal of "make the left and right treatment effects as different as possible" to the algebraic formula

Maximize:N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 
?

The answer lies in calculating the statistical variance. Here is the step-by-step mathematical proof of how they are exactly the same thing.

1. The Goal: Maximizing Variance
When we say we want the "largest difference," statistically, we mean we want to maximize the variance of the treatment effects between the new child nodes relative to the original parent node.

Let's define our terms for the parent node before the split:

N: The total number of users in the parent node (N=N 
L
​
 +N 
R
​
 ).

τ
ˉ
 : The overall Average Treatment Effect of the parent node.

We know that the parent's treatment effect is just the weighted average of the children:

N 
τ
ˉ
 =N 
L
​
  
τ
^
  
L
​
 +N 
R
​
  
τ
^
  
R
​
 
To make the children as different from each other as possible, we want to maximize how far away the children's effects ( 
τ
^
  
L
​
  and  
τ
^
  
R
​
 ) are from the parent's average ( 
τ
ˉ
 ). We write this as the weighted sum of squared differences (the variance):

Maximize:N 
L
​
 ( 
τ
^
  
L
​
 − 
τ
ˉ
 ) 
2
 +N 
R
​
 ( 
τ
^
  
R
​
 − 
τ
ˉ
 ) 
2
 
2. The Algebraic Expansion
Now, let's expand that variance equation using basic algebra (a−b) 
2
 =a 
2
 −2ab+b 
2
 :

=N 
L
​
 ( 
τ
^
  
L
2
​
 −2 
τ
^
  
L
​
  
τ
ˉ
 + 
τ
ˉ
  
2
 )+N 
R
​
 ( 
τ
^
  
R
2
​
 −2 
τ
^
  
R
​
  
τ
ˉ
 + 
τ
ˉ
  
2
 )
Next, distribute the N 
L
​
  and N 
R
​
 :

=(N 
L
​
  
τ
^
  
L
2
​
 −2N 
L
​
  
τ
^
  
L
​
  
τ
ˉ
 +N 
L
​
  
τ
ˉ
  
2
 )+(N 
R
​
  
τ
^
  
R
2
​
 −2N 
R
​
  
τ
^
  
R
​
  
τ
ˉ
 +N 
R
​
  
τ
ˉ
  
2
 )
Group the similar terms together:

=(N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 )−2 
τ
ˉ
 (N 
L
​
  
τ
^
  
L
​
 +N 
R
​
  
τ
^
  
R
​
 )+ 
τ
ˉ
  
2
 (N 
L
​
 +N 
R
​
 )
3. The Substitution (The "Aha!" Moment)
Now we apply what we know about the parent node to simplify the equation:

We know N 
L
​
 +N 
R
​
 =N

We know N 
L
​
  
τ
^
  
L
​
 +N 
R
​
  
τ
^
  
R
​
 =N 
τ
ˉ
 

Substitute those back into the equation:

=(N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 )−2 
τ
ˉ
 (N 
τ
ˉ
 )+ 
τ
ˉ
  
2
 (N)
=(N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 )−2N 
τ
ˉ
  
2
 +N 
τ
ˉ
  
2
 
Which simplifies to:

=N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 −N 
τ
ˉ
  
2
 
4. The Final Conclusion
Look at that final formula. We want to maximize:

N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 −N 
τ
ˉ
  
2
 
When the algorithm is standing at a parent node trying to decide where to draw the line, N (the total users in the node) and  
τ
ˉ
  (the parent's overall treatment effect) are already fixed. They are constants. They do not change no matter where you decide to split the data.

Because N 
τ
ˉ
  
2
  is a constant, it drops out of the optimization problem completely.

The only way to maximize the whole equation is to maximize the remaining part:

Maximize:N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 
Summary: The algorithm doesn't use this formula because it's a random guess; it uses it because it is the exact, algebraically simplified formula for statistical variance. By stripping away the parent constants, the algorithm saves massive amounts of computation time while perfectly maximizing the heterogeneity (the difference) between the left and right groups.

But what is the intuition behind breaking the data where the treatment effect is very different helping? How does it help in choosing the final prediction?

This is the perfect question to ask, because it connects the abstract math directly back to the ultimate utility of the algorithm.

The intuition boils down to this: If you do not force the algorithm to hunt for differences in treatment effect, the true causal effect gets "washed out" in the average, rendering your final predictions completely generic.

Here is the intuition behind why breaking the data this way is essential, and exactly how it leads to a highly personalized final prediction.

1. The Core Intuition: Finding "Reaction Neighborhoods"
Imagine an optimization engine like Capifi is deciding whether to show a user a premium, zero-fee loan offer (W=1) or a standard offer (W=0).

Suppose the overall Average Treatment Effect (ATE) across your entire user base is +5% (the premium offer increases overall acceptance by 5%).

Scenario A: Splitting without maximizing heterogeneity (The Washing Out Effect)
If you split the data randomly, or if you split it based purely on a variable that predicts the baseline outcome (e.g., "Has the user opened the app today?"), you might end up with two groups that both show a +5% treatment effect.

Left Group (Opened App):  
τ
^
 =+5%

Right Group (Didn't Open):  
τ
^
 =+5%

This split is useless for personalization. If you want to optimize your risk distribution, knowing that both groups react exactly the same way to the intervention gives you no actionable leverage.

Scenario B: Maximizing heterogeneity (Finding the True Signal)
Now, suppose the algorithm searches the data and splits based on a specific High, Medium, and Low (HML) risk boundary. It maximizes the variance and finds:

Left Group (Low Risk):  
τ
^
 =+15%

Right Group (High/Medium Risk):  
τ
^
 =−1%

By forcing the split to maximize the difference, the algorithm has successfully isolated a "reaction neighborhood." It discovered that Low Risk users are highly sensitive to the premium offer, while High Risk users might actually be deterred by it (or just don't care).

2. How This Helps the Final Prediction
This specific splitting strategy is what makes the final CATE prediction,  
τ
^
 (x), actually personalized. Here is how the geometry of the tree translates to the final prediction:

Creating "Pure" Leaves: Because every single split in the tree was chosen specifically to separate users by how they react to the offer, the final leaf nodes at the bottom of the tree are highly "pure." A single leaf node does not just contain users who look similar; it contains users who react to the intervention in the exact same way.

The Similarity Matchmaker: When a brand new user profile x enters the system, you drop them down all the trees in the forest. The forest's job is simply to find which historical users this new user ends up sitting next to in the leaf nodes.

Calculating the Weights (α 
i
​
 (x)): If historical user i constantly ends up in the exact same leaf nodes as new user x across hundreds of trees, the algorithm assigns historical user i a very high weight. It effectively says, "Historically, user i reacts to interventions exactly the way we expect new user x to react."

The Final Prediction: The final step is that locally weighted regression formula:

τ
^
 (x)= 
∑ 
i=1
n
​
 α 
i
​
 (x) 
W
~
  
i
2
​
 
∑ 
i=1
n
​
 α 
i
​
 (x) 
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
Because the weights α 
i
​
 (x) are only high for users in x's specific "reaction neighborhood," the final prediction  
τ
^
 (x) is computed using only the data from people who share the same underlying sensitivity to the treatment.

Summary
If the trees didn't split by maximizing treatment heterogeneity, the leaf nodes would contain a scrambled mix of users who react well and users who react poorly. The final weighted average would just spit out the generic +5% baseline for everyone.

By aggressively splitting to maximize the difference in  
τ
^
 , the Causal Forest ensures that when a new user arrives, it can isolate their specific subgroup and confidently predict, "This specific user will show a +15% uplift."

what equation is leading to this? how y and w curls are related to one another?

Those "curls" are technically called tildes ( 
Y
~
  and  
W
~
 ), and they represent residuals.

To understand how they are related, we have to look at the mathematical bridge that connects them, which is called Robinson's Transformation (or the Partially Linear Model).

Here is the exact step-by-step mathematical derivation of the equation that connects them, and how it leads to that final fraction.

1. Defining the Tildes (The Residuals)
Before looking at the equation, we need to establish what these variables actually mean. They represent the "surprise" or the "prediction error" from our baseline models.

Y
~
  
i
​
  (Outcome Residual): This is Y 
i
​
 − 
μ
^
​
 (X 
i
​
 ). It is the actual outcome minus the expected outcome.

Example: A user accepts a loan (Y=1), but based strictly on their High/Medium/Low (HML) risk profile, we only expected a 20% chance of acceptance ( 
μ
^
​
 =0.20). The residual is  
Y
~
 =+0.80. They accepted more than expected.

W
~
  
i
​
  (Treatment Residual): This is W 
i
​
 − 
e
^
 (X 
i
​
 ). It is the actual treatment received minus the expected probability of receiving it (the propensity score).

Example: A user receives the premium offer (W=1), but because they are Medium risk, historical data says they only had a 30% chance of getting it ( 
e
^
 =0.30). The residual is  
W
~
 =+0.70. They received the offer unexpectedly.

2. The Connecting Equation (Robinson's Transformation)
How are these two surprises related? We start by assuming that a user's final outcome (Y) is driven by two independent things: their baseline characteristics and the actual effect of the treatment.

We can write this as a partially linear equation:

Y 
i
​
 =f(X 
i
​
 )+τ(X 
i
​
 )⋅W 
i
​
 +ϵ 
i
​
 
(Outcome = Baseline expectation + (Treatment Effect × Treatment Received) + random noise)

To isolate the treatment effect τ(X 
i
​
 ), we apply Robinson's trick. First, we take the expected value of the entire equation, conditional on the user's features X:

E[Y 
i
​
 ∣X 
i
​
 ]=f(X 
i
​
 )+τ(X 
i
​
 )⋅E[W 
i
​
 ∣X 
i
​
 ]
Now, we subtract this expected equation from the original equation:

Y 
i
​
 −E[Y 
i
​
 ∣X 
i
​
 ]=τ(X 
i
​
 )⋅(W 
i
​
 −E[W 
i
​
 ∣X 
i
​
 ])+ϵ 
i
​
 
Notice the terms in the brackets. Y 
i
​
 −E[Y 
i
​
 ∣X 
i
​
 ] is exactly our outcome residual,  
Y
~
  
i
​
 . And W 
i
​
 −E[W 
i
​
 ∣X 
i
​
 ] is exactly our treatment residual,  
W
~
  
i
​
 .

Substitute the tildes in, and we get the fundamental relationship between them:

Y
~
  
i
​
 =τ(X 
i
​
 )⋅ 
W
~
  
i
​
 +ϵ 
i
​
 
3. Solving for the CATE
Look closely at that final equation:  
Y
~
 =τ⋅ 
W
~
 .

This is just the standard equation for a straight line (y=mx+b), where the intercept b is 0, the y-axis is  
Y
~
 , the x-axis is  
W
~
 , and the slope of the line (m) is our target: the treatment effect τ.

To find the slope of a line in statistics, we use Ordinary Least Squares (OLS) regression. We want to find the τ that minimizes the squared errors.

Because we are using a Causal Forest to make this personalized for a specific user profile x, we apply the forest's similarity weights (α 
i
​
 ) to turn it into a Weighted Least Squares problem:

Minimize: 
i=1
∑
n
​
 α 
i
​
 (x)( 
Y
~
  
i
​
 −τ 
W
~
  
i
​
 ) 
2
 
To find the minimum, you take the derivative with respect to τ and set it to zero. When you solve that derivative algebraically, you get the exact final closed-form solution:

τ
^
 (x)= 
∑ 
i=1
n
​
 α 
i
​
 (x) 
W
~
  
i
2
​
 
∑ 
i=1
n
​
 α 
i
​
 (x) 
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
The Intuition
Why go through all this math to relate the residuals instead of the raw variables? To eliminate confounding bias.

If you just looked at the raw data (Y and W), you might conclude that premium offers cause higher loan acceptance. But what if your optimization engine historically only gave premium offers to Low-risk users, who are already predisposed to accept loans anyway? The raw data lies to you.

By relating  
Y
~
  to  
W
~
 , the equation completely strips away the baseline bias. It asks a much smarter question: "When a user receives a premium offer that their risk profile didn't normally warrant ( 
W
~
  is high), do they accept the loan at a rate higher than their risk profile normally dictates ( 
Y
~
  is high)?" If both tildes move together, you have found true causality.

also how is the algorithm choosing the feature to use while splitting? is it same like building trees etc? and how does it choose alpha i when the tree is not yet fully created in the first place?

You are asking exactly the right questions to connect the theory to the actual code execution. You have zeroed in on the exact mechanics of how the algorithm searches the data, and you have spotted a crucial timing paradox regarding α 
i
​
 .

Let's break down exactly how it chooses features, and then untangle the timeline of when those weights are actually calculated.

1. How does it choose the feature to split on?
The short answer: Yes, it is almost exactly the same as building a standard Random Forest, just using the new variance formula as its compass.

Here is the exact loop the algorithm runs when it is standing at a parent node trying to decide what to do:

Random Feature Selection (Subspacing): Just like a standard Random Forest, a Causal Tree does not look at every single feature at every node. It randomly selects a small subset of features (often called mtry in programming libraries).

Example: It randomly picks "HML Risk Score," "Days Since Last Loan," and "App Opens," ignoring income and age for this specific node.

The Exhaustive Search (Greedy Splitting): The algorithm takes the first feature (e.g., Risk Score). It lines up all the users in the node from lowest to highest score. It then tests every single possible split point between those users.

Scoring: For every single split point on that feature, it calculates our variance formula: N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
 .

Repeat for all selected features: It repeats this exhaustive search for "Days Since Last Loan" and "App Opens," calculating the variance score for every possible split point on those features as well.

The Winner: The algorithm looks at all the scores across all the tested features. Whichever specific feature and specific split point produced the absolute highest variance score is chosen as the official split.

It is a "greedy" search. It doesn't plan ahead; it just looks for the single best move it can make right at that exact moment to maximize the difference in treatment effects between the left and right groups.

2. How does it choose α 
i
​
  before the tree is created?
It doesn't. You have identified a very common point of confusion.

The variable α 
i
​
 (x) represents the similarity weight between a historical training user (i) and a brand new user profile (x). Because it depends on the new user x, α 
i
​
 (x) cannot be calculated until after the entire forest is completely built and frozen.

There is a strict separation between the Training Phase and the Prediction (Inference) Phase.

Phase 1: Training (Building the Forest)
During training, the algorithm does not know or care about α 
i
​
 . It is solely focused on building the structure of the trees using the residuals ( 
Y
~
  and  
W
~
 ) and the splitting formula.

The algorithm randomly subsamples the training data.

It splits the data using the greedy search described above (the "Splitting Subsample").

It pushes the "Estimating Subsample" down the finished branches to populate the final leaf nodes with historical users.

It repeats this 500 or 1,000 times to create the forest.

At this point, training is over. The trees are frozen. No weights have been calculated yet.

Phase 2: Prediction (Calculating α 
i
​
  on the fly)
The Capifi optimization engine is running, and a brand new user x logs into the app. We need to estimate their specific CATE,  
τ
^
 (x), to decide which loan offer to show them.

Now the algorithm calculates the weights:

Drop the user down the forest: The algorithm takes new user x and drops their profile down Tree 1. Based on their features (e.g., Low Risk, 5 App Opens), they land in Leaf Node #42.

Identify the neighbors: The algorithm looks inside Leaf Node #42. Let's say there are 10 historical training users sitting in that leaf.

Repeat for all trees: The algorithm drops user x down Tree 2, Tree 3... all the way to Tree 1000.

Tally the weights (α 
i
​
 (x)): The algorithm looks back at the historical training data.

If historical user i ended up in the exact same leaf node as new user x in 800 out of 1000 trees, their α 
i
​
 (x) weight is very high (0.80).

If historical user j only shared a leaf node with user x in 5 out of 1000 trees, their α 
j
​
 (x) weight is near zero (0.005).

The Final Math: Now that it has the weights, it plugs them into the final regression equation we discussed previously, using the historical residuals of those heavily weighted "neighbor" users to calculate the personalized treatment effect for user x.

Summary: The trees are built first using simple math (maximizing variance). The trees act as a complex, data-driven "sorting hat." Later, when you need a prediction, the forest is used purely as a sorting mechanism to find out which historical users are most dynamically similar to the new user. That similarity dictates the α 
i
​
  weights.

while computing left and right tilda isn't the alpha i will be required?

This is an incredibly sharp observation. You have caught a major point of confusion in how the math is usually written in papers.

The short answer is no, α 
i
​
  is not required during the splitting phase. The confusion happens because we use the symbol  
τ
^
  for two different things at two different times. We use it for the Node-Level ATE (during splitting) and the Personalized CATE (during the final prediction).

Here is exactly how the algorithm calculates  
τ
^
  
L
​
  and  
τ
^
  
R
​
  using the residuals ( 
Y
~
  and  
W
~
 ) without needing α 
i
​
 .

1. The Splitting Phase: Unweighted Node ATE
When the tree is being built and is testing a split, it is not trying to find a personalized effect for a specific user. It is just trying to find the Average Treatment Effect (ATE) of the group of historical users inside that specific proposed node.

Let's say the algorithm is testing a split that creates a Left Node (L). Inside this proposed Left Node, there is a specific subset of our training data.

To calculate the treatment effect for this left group ( 
τ
^
  
L
​
 ), the algorithm runs a standard, unweighted Ordinary Least Squares (OLS) regression using only the residuals of the users currently sitting in that left node.

The formula looks like this:

τ
^
  
L
​
 = 
∑ 
i∈L
​
  
W
~
  
i
2
​
 
∑ 
i∈L
​
  
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
Notice two massive differences from the final prediction equation:

No α 
i
​
 : There are no weights. Every single historical user inside the left node is treated equally.

The Summation Limit: The sum is not from i=1 to n (the whole dataset). It is only summing over i∈L (the users specifically placed in the Left Node).

The algorithm calculates  
τ
^
  
L
​
  and  
τ
^
  
R
​
  using this simple, unweighted math. It plugs those numbers into the N 
L
​
  
τ
^
  
L
2
​
 +N 
R
​
  
τ
^
  
R
2
​
  formula, picks the best split, and moves on.

2. The Prediction Phase: Weighted Forest CATE
It is only after the entire forest is built that we need α 
i
​
 (x).

Now, a new user x arrives. We want to calculate their personalized CATE,  
τ
^
 (x). We use the exact same logic (relating  
Y
~
  to  
W
~
 ), but now we look at the whole dataset (i=1 to n) and apply the similarity weights we generated from the forest:

τ
^
 (x)= 
∑ 
i=1
n
​
 α 
i
​
 (x) 
W
~
  
i
2
​
 
∑ 
i=1
n
​
 α 
i
​
 (x) 
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
The "Aha!" Connection
If you want to see how perfectly these two concepts align mathematically, imagine a "forest" that consists of only one single tree.

If you drop new user x down this single tree, they land in a specific leaf node.

For all historical users i in that exact same leaf, their similarity weight is α 
i
​
 (x)= 
N 
leaf
​
 
1
​
 .

For every other historical user in the dataset, their similarity weight is α 
i
​
 (x)=0.

If you plug those specific weights into the massive, complicated Prediction Phase equation, all the users with a weight of 0 disappear. You are left only summing over the users in that specific leaf. The  
N 
leaf
​
 
1
​
  fractions on the top and bottom cancel each other out.

What are you left with?

τ
^
 (x)= 
∑ 
i∈leaf
​
  
W
~
  
i
2
​
 
∑ 
i∈leaf
​
  
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
It perfectly collapses right back into the exact same unweighted equation the algorithm used to build the node in the first place!

Summary
During splitting, the algorithm uses a localized, unweighted regression because it only cares about the people inside the proposed node. During prediction, it uses a global, weighted regression because it uses the entire forest to smoothly blend the neighborhoods together.