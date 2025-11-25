r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False-
A split is useful only if both parts are representive of the whole dataset, and
don't contain some sort of a bias.
For example, if the data is ordered such as the first part was taken in place A,
and the second part in place B, and we partition based on that, the result
might be a biased model.

2. False-
The test-set should be kept hidden from the model, until the actual testing.
This is done in order to avoid the situation the model is learning to the test
specifically, thus not getting the same results we would get in the real world,
where the model has no prior knowledge of the data.

3. True-
This is a crucial part in the cross-validation method.
We use the score in each fold, than take the average of all folds in order to 
gauge a score of how good the model is, as we have had different sets of
training data and test in each fold- so the average score is less impacted by
specific fold noise.

4. True-
if the model tends to overfit into the data, introducing some label noise might
reveal this tendency. This is because it would try to latch on the incorrect
information, thus when the test comes, it will fail much more, as it will
this time miss-identify the correct labels (as it overffited to wrong data).


"""

part1_q2 = r"""
**Your answer:**
No-
His method trains on the test data, thus introducing data leakage, by
fine-tuning $\lambda$ to be the best fitted to this specific test data.
Now there is the risk $\lambda$ is overffitted to the test set, and not generalised.


"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your Answer:**

If we allow $\Delta < 0$, the SVM attempts to enforce a "negative margin," which breaks the classification logic.

Recall the condition for incurring zero loss on a specific sample $i$ and incorrect class $j$:
$$
\max(0, \Delta + s_j - s_{y_i}) = 0 \iff s_{y_i} \ge s_j + \Delta
$$

If $\Delta$ is negative (e.g., $\Delta = -5$), this condition becomes:
$$
s_{y_i} \ge s_j - 5
$$

**The consequences are:**
1.  **Permitted Misclassification:** The model can predict a score for the correct class ($s_{y_i}$) that is **lower** than the score for an incorrect class ($s_j$) and still incur **zero loss**. 
2.  **Failure to Learn:** Since the optimization goal is to minimize loss, the model is no longer incentivized to push the correct class score *above* the others. It is satisfied even if the correct class is "close enough" to the wrong one, or even slightly below it.
3.  **Trivial Solutions:** Combined with the regularization term $\frac{\lambda}{2} \|W\|^2$, the model will likely drive the weights towards zero. If the weights are small enough, differences in scores become small, and the condition $s_{y_i} \ge s_j + \Delta$ (where $\Delta$ is negative) is easily satisfied without actually separating the data.
"""

part2_q2 = r"""
**Your answer:**

1. **Interpretation of what the model learns:**
   The linear model is essentially learning a **"template"** or **"prototype"** for each class. 
   Since the score is calculated as a dot product $s_j = \mathbf{w}_j^T \mathbf{x} + b_j$, the weights $\mathbf{w}_j$ act as a filter:
   * **Positive weights (Bright pixels):** Represent regions where the model expects to see "ink" (signal) for that specific digit. If the input image also has ink there, the score increases.
   * **Negative weights (Dark pixels):** Represent regions where the model expects background (no ink). If the input image has ink there, the score decreases (penalizing the prediction).
   Visually, the weight images look like blurry, "ghostly" averages of the digits in the training set.

2. **Explanation of Classification Errors:**
   Because the model is strictly linear, it has significant limitations:
   * **No Translation Invariance:** The model relies on pixels being in specific fixed locations. If a digit is shifted slightly to the left or right, or written with a different slant, it may no longer align with the learned "template," leading to a low score.
   * **Overlapping Structures:** Digits that share similar pixel distributions are easily confused. For example, an '8' and a '3' share the same top and bottom loops. If a specific '3' is written widely, it might trigger the positive weights of the '8' template enough to cause a misclassification. The linear model looks at the sum of pixel intensities and cannot understand complex geometric structures (like "this loop is closed" vs "this loop is open").
"""

part2_q3 = r"""
**Your answer:**

1. **Learning Rate: Good**
   The loss graph shows a rapid decrease in the initial epochs and then gradually plateaus (stabilizes) towards a minimum value. This indicates the step size was sufficient to descend the gradient efficiently without instability.
   
   **Comparison to other cases:**
   * **If the Learning Rate were Too Low:** The loss graph would decrease very slowly and linearly. After the same number of epochs, the loss would still be high and far from the minimum (it would look like a straight line pointing down that hasn't flattened out yet).
   * **If the Learning Rate were Too High:** The loss graph would look unstable ("jittery"). The loss might decrease very sharply at the very beginning but then fluctuate wildly or bounce around a high value without settling. in extreme cases, the loss might even increase (diverge) over time.

2. **Model Fit: Slightly overfitted to the training set**
   We observe that the training accuracy is consistently higher than the validation/test accuracy. This gap indicates that the model has learned some specific noise or patterns in the training data that do not generalize perfectly to unseen data.
   However, the gap is likely small (e.g., a few percentage points), and the test accuracy is still relatively high for a linear model. This suggests the overfitting is only "slight" rather than "highly overfitted" (where we would see training accuracy near 100% and test accuracy very low).
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

1. **Ideal Pattern:**
   While there is no specific "shape" we look for, the ideal scenario is to see **random noise** centered around 0. The residuals should be a cloud of points with no discernable trends or patterns. 
   If we see a clear pattern (e.g., a line or curve), it implies that our model failed to capture some signal in the data (systematic error) or that the data is not linear. If we see a "fanning out" shape, it implies the variance is not constant.

2. **Fitness of the trained model:**
   Our model appears to be **quite well fitted**. 
   We do not observe any distinct geometric patterns in the residual plot (residuals appear random), suggesting the linear model captured the main relationships in the data. Additionally, the MSE difference between the training and test sets is not particularly high, indicating that we are not suffering from significant overfitting.

3. **Comparison (Top-5 vs. Final):**
   Comparing the final graph to the top-5 features graph, the **loss is significantly lower** and the residuals are more tightly clustered around zero.
   This improvement occurs because the final model was trained using **Cross-Validation** to select the optimal hyperparameters and feature set, whereas the previous model was artificially restricted to only the top-5 features. This allowed the final model to capture more information and generalize better.
"""

part3_q2 = r"""
**Your answer:**

1. **Yes, it is still a linear regression model.** Linear regression is defined by the relationship being linear with respect to the **model parameters** (the weights $\mathbf{w}$), not necessarily the input features.
   If we transform our input $\mathbf{x}$ using a non-linear function $\Phi(\mathbf{x})$, our model becomes $y = \mathbf{w}^T \Phi(\mathbf{x}) + b$. Since $y$ is still a linear combination of the parameters $\mathbf{w}$, it remains a linear model. We can still use the exact same techniques (like Least Squares or Gradient Descent) to find the optimal $\mathbf{w}$.

2. **No.**
   We can only fit functions that can be represented as a **linear combination** of the specific non-linear features we added.
   For example, if we only add polynomial features (like $x^2, x^3$), we can fit any polynomial function, but we cannot perfectly fit a sine wave, an exponential function, or a step function (unless we add those specific transforms as features). We are limited to the subspace spanned by our chosen basis functions.

3. **Effect on the decision boundary:**
   * **In the feature space:** The decision boundary remains a hyperplane. The model is still separating the data linearly in the high-dimensional space defined by the non-linear features ($\mathbf{w}^T \Phi(\mathbf{x}) + b = 0$).
   * **In the original input space:** The decision boundary will **no longer be a hyperplane**. It will become a non-linear surface (curved, circular, complex shapes) corresponding to the projection of that high-dimensional hyperplane back into the original dimensions. This allows the linear classifier to solve problems that are not linearly separable in the original space (like the XOR problem).
"""

part3_q3 = r"""
**Your answer:**

1.
$
\begin{aligned}
\mathbb{E}_{x,y}\big[|y-x|\big]
&= \int_{0}^{1}\int_{0}^{1} |x-y| \,dy\,dx \\[6pt]
&= 2\int_{0}^{1}\int_{0}^{x} (x-y)\,dy\,dx \\[6pt]
&= 2\int_{0}^{1}\left[ xy-\frac{y^2}{2} \right]_{y=0}^{y=x} dx \\[6pt]
&= 2\int_{0}^{1}\left(x^2-\frac{x^2}{2}\right)dx \\[6pt]
&= 2\int_{0}^{1}\frac{x^2}{2}\,dx \\[6pt]
&= \int_{0}^{1} x^2\,dx \\[6pt]
&= \left[\frac{x^3}{3}\right]_{0}^{1} \\[6pt]
&= \frac{1}{3}.
\end{aligned}
$

2.
$
\begin{aligned}
\mathbb{E}_x\big[|\hat{x}-x|\big] 
&= \int_0^1 |\hat{x}-x| \, dx \\[2mm]
&= \int_0^{\hat{x}} (\hat{x}-x)\, dx + \int_{\hat{x}}^1 (x-\hat{x})\, dx \\[1mm]
&= \left[\hat{x}x - \frac{x^2}{2}\right]_0^{\hat{x}} + \left[\frac{x^2}{2} - \hat{x}x\right]_{\hat{x}}^1 \\[1mm]
&= \left(\hat{x}^2 - \frac{\hat{x}^2}{2}\right) + \left(\frac{1}{2} - \hat{x} - \left(\frac{\hat{x}^2}{2}-\hat{x}^2\right)\right) \\[1mm]
&= \left(\frac{\hat{x}^2}{2}\right) + \left(\frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}\right) \\[1mm]
&= - \hat{x} + \hat{x}^2 + \frac{1}{2}
\end{aligned}
$

3.
The scalar term ($+\frac{1}{2}$) in the polynomial derived above is a constant with respect to the learnable parameter $\hat{x}$. 
When we train the model, we minimize this loss function using gradient-based optimization. The derivative of a constant is zero ($\nabla_{\hat{x}} C = 0$). Therefore, the constant term does not affect the gradient, nor does it change the specific value of $\hat{x}$ that minimizes the function (the $\text{argmin}$).
"""

# ==============

# ==============
