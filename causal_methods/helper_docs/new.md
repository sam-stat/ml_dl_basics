do the depth of the causal trees are always 1?

No, the depth of a Causal Tree is almost never 1. A tree with a depth of 1 is called a "decision stump." While algorithms like AdaBoost use stumps, Causal Forests rely on growing deep trees.

Here is why they need to be deep, and the specific rule that forces them to eventually stop growing.

1. Why they grow deep: Discovering Complex "AND" Logic
If a tree only had a depth of 1, it could only split the data on a single feature. For example, it could only say: "Split users by Risk Score > 60."

But human behavior is rarely that simple. In a lending optimization engine like Capifi, a user's reaction to a premium offer usually depends on a complex combination of factors.

You might find that the premium offer is highly effective only if:

(Depth 1): Risk is Low... AND

(Depth 2): They have opened the app in the last 7 days... AND

(Depth 3): They have previously taken a loan for a medical emergency.

To discover these highly specific "reaction neighborhoods," the tree must grow deep, branching on different features sequentially to carve out the exact subgroup that responds best to the intervention.

2. When do they stop? The "Treatment Overlap" Constraint
While standard Random Forest trees can theoretically grow until there is only 1 person left in a leaf, Causal Trees cannot do this. They have a strict, mathematical stopping mechanism.

Think back to the unweighted regression formula the tree uses to calculate the treatment effect inside a node:

τ
^
 = 
∑ 
W
~
  
i
2
​
 
∑ 
Y
~
  
i
​
  
W
~
  
i
​
 
​
 
To calculate that math, you must have both treated users (W=1) and control users (W=0) inside the leaf. If the tree grows so deep that a leaf node only contains 3 people, and by pure chance all 3 of them received the standard offer, the variance of the treatment goes to zero. You cannot calculate the difference between treatment and control if the control group doesn't exist in that leaf!

The Stopping Rule:
When you configure a Causal Forest (in libraries like grf or EconML), you must set a hyperparameter usually called min_node_size or min_treatment_size.

The tree will keep growing deeper and deeper until it hits a point where splitting again would result in a leaf that doesn't have enough treated or control users to run a statistically significant regression (usually a minimum of 5 to 10 users of each type per leaf).

Summary:
Causal trees are grown as deep as possible to find highly personalized, multi-variable subgroups, but they are forced to stop growing the moment they risk running out of the treatment/control overlap needed to actually measure the causal effect.

how does honesty work when we are growing deep trees?

This is a brilliant follow-up because growing deep trees is exactly where the concept of "Honesty" does its heavy lifting.

When people first hear about Honest Trees, they often assume the algorithm cuts the data in half at every single split as it goes deeper. If it did that, you would run out of data almost instantly (e.g., 1000 users → 500 → 250 → 125 → 62).

That is not how it works. The split happens only once per tree, right at the beginning.

Here is exactly how Honesty is maintained as the tree grows deep, and why it is the ultimate shield against overfitting.

1. The Timeline of an Honest Deep Tree
Let’s trace the life of a single tree in the forest. For this specific tree, the algorithm subsamples 5,000 historical users from your Capifi dataset.

Step 1: The Root Split (The Halving)
Before the tree even starts growing, the algorithm randomly divides those 5,000 users into two completely separate buckets:

The Splitting Set (J 
1
​
 ): 2,500 users.

The Estimating Set (J 
2
​
 ): 2,500 users.

Step 2: Growing the Structure (Using ONLY J 
1
​
 )
The algorithm takes the J 
2
​
  bucket, sets it aside, and completely ignores it. It uses only the 2,500 users in J 
1
​
  to build the tree.
It looks at J 
1
​
 , maximizes the variance (N 
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
 ), and splits the root. It looks at the resulting left node (which now has, say, 1,200 J 
1
​
  users), finds the best split, and branches again. It keeps doing this, growing deeper and deeper until it hits that stopping rule (minimum node size) we discussed earlier.

Crucially, the tree structure might heavily overfit to the quirks of the J 
1
​
  users. Because it grew deep, it might have created a hyper-specific leaf node for "Low Risk users who opened the app 3 times and borrowed exactly $100 last month," purely because 8 people in J 
1
​
  fit that description and had a weirdly high treatment effect.

Step 3: Populating the Leaves (Using ONLY J 
2
​
 )
Once the tree has reached its maximum depth, the branches are frozen. The J 
1
​
  users are now thrown away.

The algorithm brings back the 2,500 users in the J 
2
​
  Estimating Set. It drops them down the frozen branches of the tree.

Remember that hyper-specific leaf node the tree created? When we drop the J 
2
​
  users down, we might find that zero J 
2
​
  users land in that leaf.

Or, we might find that 5 users land in it, but their actual treatment effect is completely average, not wildly high like the J 
1
​
  users were.

2. Why this is magic for Deep Trees
In a standard, "dishonest" decision tree, if you let it grow very deep, it will memorize the training data. If you use the same people to draw the boundaries AND calculate the final value in the leaf, your confidence in that final value is artificially inflated. You are just measuring the noise.

Honesty breaks this loop. Because J 
2
​
  had absolutely zero influence on how the boundaries were drawn, any treatment effect calculated using J 
2
​
  is mathematically independent of the tree's structure.

If the deep branches of the tree found a genuine causal rule (e.g., Low Risk users really do react better to premium offers), then the J 
2
​
  users who fall into that leaf will also show a high treatment effect. The signal survives.
If the deep branches just memorized random noise in J 
1
​
 , the J 
2
​
  users will act as a reality check, and the calculated effect in that leaf will safely regress back to the baseline.

3. The Trade-Off: The Data Hunger
There is no free lunch in statistics. While Honesty saves deep trees from overfitting, it introduces a severe constraint: It cuts your effective sample size in half.

Because you need enough J 
2
​
  users (both treated and control) to land in every single deep leaf node to calculate the regression formula ( 
τ
^
 = 
∑ 
W
~
  
i
2
​
 
∑ 
Y
~
  
i
​
  
W
~
  
i
​
 
​
 ), Causal Forests require significantly more data than standard Random Forests.

If your Capifi dataset is small, cutting it in half for Honesty means your trees cannot grow very deep before the J 
2
​
  leaves become too empty to calculate a statistically significant treatment effect. This is why Causal Forests shine brightest in environments with large, rich datasets where sacrificing half the rows to guarantee "honesty" is a price worth paying for highly accurate personalization.