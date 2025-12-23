r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

**1. Shape of the Jacobian $\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}$**

The Jacobian contains the partial derivative of every element in the output tensor $\mathbf{Y}$ with respect to every element in the input tensor $\mathbf{X}$.
* Shape of $\mathbf{Y}$: $(N, D_{out}) = (64, 512)$
* Shape of $\mathbf{X}$: $(N, D_{in}) = (64, 1024)$

The resulting Jacobian shape is the concatenation of these shapes:
$$(N, D_{out}, N, D_{in}) \rightarrow (64, 512, 64, 1024)$$


**2. Structure of the Jacobian**

If we view the Jacobian as a 2D matrix where the rows correspond to the flattened output and columns to the flattened input, it has a **block diagonal structure**.

This is because the output for a specific sample $i$, denoted $\mathbf{y}_i$, depends *only* on the input for that same sample $\mathbf{x}_i$. It is independent of any other sample $\mathbf{x}_j$ (where $j \neq i$).
* **Diagonal blocks:** Non-zero blocks of shape $(D_{out} \times D_{in})$. Each block represents the local derivative $\frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_i}$ for one sample.
* **Off-diagonal blocks:** All zeros, because $\frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_j} = 0$ for $i \neq j$.


**3. Optimization**

**Yes.** We do not need to store the full Jacobian.
Because the layer is linear and the weights are shared across the batch, the local derivative $\frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_i}$ is identical for every sample $i$. Specifically, $\frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_i} = \mathbf{W}$.

Therefore, instead of storing a sparse tensor with $N$ copies of the weights, we only need to store the weight matrix itself.
* **New tensor shape:** $(D_{out}, D_{in}) \rightarrow (512, 1024)$.


**4. Calculating $\delta \mathbf{X}$**

We can calculate the gradient w.r.t. the input using matrix multiplication, leveraging the chain rule without forming the large Jacobian. Given the relationship $\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$:

$$ \delta \mathbf{X} = \delta \mathbf{Y} \cdot \mathbf{W} $$

Dimensional analysis:
$$ (N, D_{in}) = (N, D_{out}) \times (D_{out}, D_{in}) $$
$$ (64, 1024) = (64, 512) \times (512, 1024) $$


**5. Shape of Jacobian $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$**

The Jacobian represents the derivatives of the output $\mathbf{Y}$ (shape $N, D_{out}$) with respect to the weights $\mathbf{W}$ (shape $D_{out}, D_{in}$).

* **Tensor Shape:** $(N, D_{out}, D_{out}, D_{in}) \rightarrow (64, 512, 512, 1024)$.

If we view this as a block matrix mapping vectorized weights to vectorized output, it consists of **$D_{out}$ blocks** (512 blocks) along the diagonal.
* **Block Shape:** Each block corresponds to the input matrix $\mathbf{X}$.
* **Dimensions:** $(N \times D_{in}) = (64 \times 1024)$.
"""

part1_q2 = r"""
**Your answer:**

**Yes, the second-order derivative can be helpful.**

While standard gradient descent relies only on the first derivative (gradient) to determine the *direction* of the step, the second derivative, the Hessian matrix, provides information about the curvature of the loss.

**Scenario where it is useful: Ill-conditioned Landscapes (Ravines)**

Imagine a loss landscape shaped like a narrow, elongated valley (a "ravine").
* **Gradient Descent (1st Order):** The gradient is very steep perpendicular to the ravine walls but shallow along the valley floor. Standard gradient descent tends to oscillate ("zig-zag") back and forth across the ravine walls, making very slow progress toward the minimum.
* **Newton's Method (2nd Order):** By using the inverse Hessian matrix ($\mathbf{H}^{-1}$), the optimizer can normalize the curvature. It essentially "rescales" the landscape to look more spherical, allowing it to take a direct path toward the minimum. This leads to **quadratic convergence** (much faster than the linear convergence of gradient descent).

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.05
    reg = 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1 
    lr_vanilla = 0.02  
    lr_momentum = 0.005 
    lr_rmsprop = 0.0002 
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 5e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

**1. No-dropout (Blue) vs. Dropout (Orange/Green)**

**Yes, the graphs match the expected behavior of dropout regularization.**

* **No-dropout (`dropout=0` - Blue line):**
    * **Behavior:** This model exhibits overfitting. It achieves very high training accuracy (>80%) and very low training loss, indicating it has memorized the training data.
    * **Evidence:** In the `test_loss` graph, the blue line starts low but then increases dramatically as training progresses. This divergence between the decreasing training loss and increasing test loss points to overfitting. The model is becoming less generalizable as it continues to train.

* **With Dropout (`dropout=0.4` - Orange line):**
    * **Behavior:** The dropout Makes it harder for the model to memorize the training data.
    * **Evidence:** The training accuracy is significantly lower (~45%) compared to the no-dropout model, but the **test loss is stable** and does not explode like the blue line. Crucially, the `test_acc` (bottom right) for the orange line eventually surpasses the blue line, showing that the model generalizes better to unseen data despite performing worse on the training set.

**2. Low-dropout (0.4) vs. High-dropout (0.8)**

The graphs demonstrate the trade-off between regularization and model capacity:

* **Low Dropout (`dropout=0.4` - Orange):** This setting strikes a good balance. It provides enough noise to prevent the overfitting seen in the blue line (keeping test loss flat) while still allowing enough information to pass through for the network to learn patterns (training accuracy rises to ~45%).
* **High Dropout (`dropout=0.8` - Green):** This setting is **too aggressive** and leads to **underfitting**.
    * **Evidence:** In the `train_loss` graph, the green line barely decreases, and in the `train_acc` graph, it hovers near 10-15% (which is close to random guessing for 10 classes). By dropping 80% of the neurons, the effective capacity of the network is reduced so drastically that it cannot learn the underlying function of the data at all.
"""

part2_q2 = r"""
**Your answer:**

**Yes, it is possible.**

It is possible for the test loss to decrease while test accuracy decreases because they measure different things:
* **Accuracy** only cares about which class has the highest score. It does not matter if the correct class wins by 0.01 or by 0.99.
* **Cross-Entropy Loss** cares about the confidence (probability) assigned to the correct class.

**How it happens:**
Imagine a scenario where the model is evaluated on a test set:
1.  **Improving Confidence on Easy Samples:** For the majority of samples that the model is already classifying correctly, the model becomes much more confident (e.g., the probability of the correct class rises from 0.6 to 0.99). This causes a big drop in the total cross-entropy loss.
2.  **failing borderline samples:** Simultaneously, a few borderline samples that were previously barely correct (e.g., probability 0.51) shift slightly to become barely incorrect (e.g., probability 0.49). This causes the accuracy to drop.

In this case, the significant reduction in loss from step #1 outweighs the small increase in loss from step #2, resulting in a **lower average loss** even though the **count of correct predictions (accuracy)** has gone down.
"""

part2_q3 = r"""
**Your answer:**

**1. GD vs. SGD**

* **Similarities:**
    * Both are iterative optimization algorithms used to minimize a loss function $L(\theta)$.
    * Both update parameters $\theta$ by moving in the opposite direction of the gradient: $\theta_{t+1} = \theta_t - \eta \nabla L$.
    * Both require a differentiable loss function and hyperparameters like learning rate ($\eta$).

* **Differences:**
    * **Data Usage:** GD computes the gradient using the **entire dataset** for every single update step. SGD computes the gradient using only a **single sample** (or a small batch) per step.
    * **Computation:** GD is computationally expensive per step (slow updates) but provides a stable, deterministic path. SGD is very fast per step (frequent updates) but follows a noisy, zigzagging path.
    * **Convergence:** GD converges linearly to a local minimum. SGD oscillates around the minimum due to noise but can escape shallow local minima more easily because of that same noise.

**2. Momentum in GD**

**Yes, you should incorporate momentum into GD.**

While momentum is famous for dampening the noise in SGD, it also solves a critical problem in full-batch GD: **poor conditioning (Ravines)**.
In GD, if the loss landscape has a steep slope in one direction and a shallow slope in another (a ravine), standard GD will oscillate back and forth across the steep sides while making tiny progress along the shallow valley floor. Momentum accumulates velocity along the consistent direction (the valley floor) and cancels out the oscillations across the steep walls, significantly accelerating convergence even without the noise of SGD.

**3. Simulating GD with Batches**

**3.1. Equivalence to GD**
**Yes, it is equivalent (assuming linearity of the derivative).**
The loss function for the full dataset is usually the sum (or average) of the losses of individual samples: $L_{total} = \sum L_i$.
Since the derivative is a linear operator, the gradient of the sum is the sum of the gradients:
$$ \nabla \left( \sum_{i=1}^N L_i(\theta) \right) = \sum_{i=1}^N \nabla L_i(\theta) $$
Therefore, accumulating the gradients (or losses) from disjoint batches and stepping once is mathematically identical to calculating the gradient over the whole dataset at once.

**3.2. Out of Memory Error**
The error occurred because we tried to do **one backward pass on the sum of the losses**.
To track the backpropagation, we use a **computational graph**. By feeding batch after batch and summing the losses without detaching or backwarding, the framework keeps extending this graph in memory, storing all intermediate activations for **all** batches to eventually compute gradients. This effectively reconstructs the memory footprint of the full dataset, defeating the purpose of splitting it.

**3.3. Solution**
Instead of summing the *losses* and doing one backward pass, we should **accumulate the gradients**.
1.  For each batch, run the forward pass and calculate loss.
2.  Run `loss.backward()` immediately. This computes gradients and frees the graph for that batch.
3.  Accumulate these gradients in the parameter `.grad` attributes (PyTorch does this by default if you don't call `zero_grad()`).
4.  Repeat for all batches.
5.  Call `optimizer.step()` only after processing all batches.
6.  Call `optimizer.zero_grad()`.
"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 2
    hidden_dims = 32
    activation = "relu"
    out_activation = "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    
    lr = 0.01
    weight_decay = 1e-3
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

**1. Definitions of Error Types**

* **Optimization Error:**
    This is the difference between the empirical loss of the specific model our algorithm found, $h$, and the minimum possible empirical loss achievable by any model in our hypothesis class, $h^*$:
    $$ \epsilon_{opt} = L_{train}(h) - L_{train}(h^*) $$
    It measures the failure of the optimization algorithm (e.g., SGD getting stuck in a local minimum or converging too slowly).

* **Generalization Error (Estimation Error):**
    This is the difference between the expected loss (true risk) on unseen data and the empirical loss on the training data:
    $$ \epsilon_{gen} = L_{test}(h) - L_{train}(h) $$
    It measures how well the model applies what it learned to new data (i.e., the degree of **overfitting**).

* **Approximation Error:**
    This is the difference between the best possible model in our hypothesis class, $h^*$, and the optimal model theoretically possible (Bayes optimal), $h_{bayes}$:
    $$ \epsilon_{app} = L(h^*) - L(h_{bayes}) $$
    It measures the limitations of the model architecture itself (i.e., underfitting due to a lack of capacity).

---

**2. Qualitative Analysis of the Model**

Based on the loss and accuracy plots provided (specifically the second set of graphs with unstable test metrics):

1.  **Optimization Error: Low**
    * **Observation:** The training loss (blue line in `test_loss`) decreases smoothly and consistently, stabilizing at a relatively low value (about 0.15). The training accuracy reaches a high value (about 94%).
    * **Conclusion:** The optimizer successfully minimized the objective function on the training set. There is no sign of getting stuck at a high loss value.

2.  **Generalization Error: High**
    * **Observation:** There is a significant and erratic gap between the training curves (blue) and the test curves (orange). While the training loss is low and stable, the test loss spikes repeatedly and remains much higher. Similarly, the test accuracy fluctuates wildly (between 75% and 92%) compared to the stable ~94% training accuracy.
    * **Conclusion:** The model is **overfitting**. It performs significantly worse on unseen data than on the data it was trained on. I would definitely take measures to decrease this (e.g., adding Dropout, L2 regularization, or more data).

3.  **Approximation Error: Low**
    * **Observation:** The model achieves very high accuracy (~94%) on the training data.
    * **Conclusion:** The hypothesis class (the MLP architecture) is sufficiently complex to capture the underlying patterns in the data. If the approximation error were high, the model would fail to fit even the training set (high bias/underfitting), which is not the case here.
"""

part3_q2 = r"""
**Your answer:**

**1. Optimize for Low FPR (Minimize False Positives)**
* **Goal:** We want to be very sure before predicting "Positive". We are willing to miss some real positives to avoid crying wolf.
* **Scenario: Spam Email Filter**
    * **False Positive (High Cost):** A legitimate, important email (e.g., a job offer or a message from family) is classified as spam and hidden in the junk folder. The user never sees it, potentially causing significant personal or professional harm.
    * **False Negative (Low Cost):** A spam email lands in the main inbox. The user simply deletes it. It is a minor annoyance but not a disaster.
    * **Conclusion:** In this case, we enforce a strict threshold to keep FPR very low, even if it means the FNR increases (some spam gets through).

**2. Optimize for Low FNR (Minimize False Negatives)**
* **Goal:** We want to catch every single "Positive" case. We are willing to deal with false alarms to ensure nothing is missed.
* **Scenario: Screening for a Deadly Disease (e.g., Cancer or COVID-19)**
    * **False Negative (High Cost):** A sick patient is diagnosed as healthy. They are sent home without treatment, leading to worsening health or spreading the infection to others. This outcome is potentially fatal.
    * **False Positive (Low Cost):** A healthy patient is flagged as potentially sick. They undergo further, more precise testing (like a biopsy or PCR). This causes temporary anxiety and financial cost, but the patient remains safe.
    * **Conclusion:** In this case, we prefer a "sensitive" model with a very low FNR to ensure safety, accepting a higher FPR (more false alarms) as a necessary trade-off.
"""

part3_q3 = r"""
**Your answer:**

**1. Fixed Depth, Width Varies (Columns)**
* **Observation:** Looking at the columns (e.g., the left-most column with `depth=1` or right-most with `depth=4`), we see that increasing the `width` significantly improves the model's ability to fit the data.
    * **Small Width (width=2):** The model heavily underfits. The decision boundary is overly simplistic (nearly linear or simple corners) because the information is compressed into a 2-dimensional bottleneck at every layer, losing the necessary dimensionality to disentangle the non-linear "moons".
    * **Large Width (width=32):** The model produces a smooth, confident boundary that fits the crescent shapes well.
* **Explanation:** Width corresponds to the dimensionality of the feature space. By increasing width, we give the network more "neurons" to act as basis functions. This allows the model to project the data into a higher-dimensional space where the classes are more easily separable, resulting in a flexible and smooth decision boundary.

**2. Fixed Width, Depth Varies (Rows)**
* **Observation:** Looking at the rows (e.g., the middle row with `width=8`), increasing `depth` changes the complexity of the boundary texture.
    * **Shallow (depth=1):** The boundary is very smooth and simple.
    * **Deep (depth=4):** The boundary becomes more intricate, "wiggly," and sharp. For the narrow case (`width=2`), increasing depth actually hurt performance (vanishing gradients or bottleneck issues), but for `width=8` and `32`, it allowed for finer adjustments to the boundary.
* **Explanation:** Depth allows the model to learn hierarchical features and compose simple non-linearities into highly complex functions. While a shallow network learns a smooth global approximation, a deep network constructs the boundary piecewise, leading to sharper transitions and the ability to capture more detailed/high-frequency patterns in the data.

**3. Comparison: `depth=1, width=32` vs. `depth=4, width=8`**
* **Shallow & Wide (`depth=1, width=32`):** (Bottom Left)
    * **Boundary:** Very smooth, broad curvature.
    * **Performance:** High accuracy (~89.4%).
    * **Mechanism:** Acts as a universal approximator using many simple basis functions in parallel. It captures the global trend well but lacks "sharpness."
* **Deep & Narrow (`depth=4, width=8`):** (Top Right)
    * **Boundary:** Sharper, more segmented/intricate boundary.
    * **Performance:** Slightly higher accuracy (~90.2%).
    * **Mechanism:** Uses composition of functions to approximate the complex topology efficiently. It fits the specific "wiggles" of the noise better than the smooth wide model.
* **Conclusion:** The Deep & Narrow model is more parameter-efficient for capturing complex non-linearities (higher accuracy with similar computational budget), but the Wide & Shallow model produces a smoother, "safer" boundary that may be more robust to noise in some regions.

**4. Threshold Selection Effect**
* **Did it improve results?** Yes, it likely improved (or at least maintained) the results compared to a default threshold of 0.5.
* **Why?** Neural networks trained with Cross-Entropy loss output probabilities, but these probabilities are not always perfectly calibrated (e.g., the model might be "timid," outputting max probabilities of 0.45 for a positive class). If we used a fixed threshold of 0.5, we might classify everything as negative (Accuracy = 50%).
    By selecting the threshold on the **validation set** (using ROC analysis to maximize the TPR-FPR difference), we find the optimal operating point that separates the classes given the model's actual output distribution. This effectively "calibrates" the final binary decision, maximizing accuracy on unseen test data.
"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**1. Number of parameters**

Let's calculate the parameters (weights) for each case, assuming bias is ignored (standard for ResNet with BatchNorm) and the bottleneck reduces dimensionality by a factor of 4 (256 -> 64).

* **Regular Block:** Two $3 \times 3$ convolutions with 256 input and output channels.
    $$ 2 \times (3 \cdot 3 \cdot 256 \cdot 256) = 2 \times 589,824 = \mathbf{1,179,648} $$

* **Bottleneck Block:** Three convolutions: $1\times 1$ (256 to 64), $3 \times 3$ (64 to 64), and $1 \times 1$ (64 to 256).
    1.  Projection (reduce): $1 \cdot 1 \cdot 256 \cdot 64 = 16,384$
    2.  Processing: $3 \cdot 3 \cdot 64 \cdot 64 = 36,864$
    3.  Expansion (restore): $1 \cdot 1 \cdot 64 \cdot 256 = 16,384$
    $$ 16,384 + 36,864 + 16,384 = \mathbf{69,632} $$

The bottleneck block has significantly fewer parameters (approx. 17x fewer).

**2. Number of floating point operations (FLOPs)**

**Qualitative Assessment:** The number of FLOPs is generally proportional to the number of parameters multiplied by the spatial resolution of the feature map ($H \times W$). Since both blocks operate on the same spatial resolution, the large reduction in parameters in the Bottleneck block leads to a proportional, massive reduction in FLOPs. This makes the bottleneck much more computationally efficient.

**3. Ability to combine input**

1.  **Spatially (within feature maps):** The **Regular Block** has better spatial combination ability because it applies two $3\times3$ filters sequentially, effectively increasing the receptive field more than the Bottleneck block, which only has a single $3\times3$ convolution in the middle (the $1\times1$ layers do not look at neighboring pixels).
2.  **Across feature maps:** The **Bottleneck Block** is specifically designed to manipulate cross-channel information efficiently. The $1\times1$ layers perform linear combinations of the input channels to compress and then expand the depth. While the Regular block also combines features across maps (as every standard convolution does), the Bottleneck decouples this operation, allowing for complex cross-channel mixing with much less computation.
"""


part4_q2 = r"""
**1. Derivation for $y_1$**

Given $y_1 = M \cdot x_1$, the derivative with respect to $x_1$ using the chain rule is:
$$ \frac{\partial L}{\partial x_1} = \left(\frac{\partial y_1}{\partial x_1}\right)^T \frac{\partial L}{\partial y_1} = \mathbf{M^T \frac{\partial L}{\partial y_1}} $$

**2. Derivation for $y_2$**

Given $y_2 = x_2 + M \cdot x_2 = (I + M)x_2$:
$$ \frac{\partial L}{\partial x_2} = \left(\frac{\partial y_2}{\partial x_2}\right)^T \frac{\partial L}{\partial y_2} = (I + M)^T \frac{\partial L}{\partial y_2} = (I + M^T) \frac{\partial L}{\partial y_2} $$
$$ = \mathbf{\frac{\partial L}{\partial y_2} + M^T \frac{\partial L}{\partial y_2}} $$

**3. Explanation of Vanishing Gradients**

In deep networks, the gradient at an early layer is the product of the Jacobians of all subsequent layers (Chain Rule).
* In the standard case ($y_1$), if we have many layers where weights are small ($|M_{i,j}| < 1$), the backpropagated gradient involves a product of many matrices $M^T$. This product tends to decay exponentially toward zero, causing the **vanishing gradient** problem.
* In the residual case ($y_2$), the gradient signal is multiplied by $(I + M^T)$. Even if $M$ has small entries, the term $(I + M^T)$ is close to the Identity matrix. This allows the gradient $\frac{\partial L}{\partial y_2}$ to flow backwards through the "skip connection" (the $+ \frac{\partial L}{\partial y_2}$ term) without being diminished, effectively creating a "gradient superhighway" that enables training of very deep networks.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**1. Effect of depth on accuracy:**
We can see that "deeper" is definitely not always better. The accuracy actually drops significantly as we add layers. The best results were obtained with $L=4$ (reaching around 70% accuracy). Deeper networks like $L=8$ and $L=16$ failed completely. This happens because deeper networks are much harder to optimize; the gradient has to travel through many more layers during backpropagation, making the signal weaker and unstable.

**2. Untrainable depths:**
Yes, for $L=8$ and $L=16$ the network was effectively not trainable at all, staying stuck at ~10% accuracy (which is just random guessing for 10 classes).
This is caused by the **vanishing gradient problem**. The gradients become vanishingly small as they propagate back through the deep layers, so the weights in the early layers effectively never get updated.

Two ways to resolve this:
1. **Residual Connections (ResNet):** Adding skip connections allows the gradient to flow directly through the network, preventing them from vanishing.
2. **Batch Normalization:** Normalizing the layer inputs ensures that activations and gradients remain in a stable range, which makes training deep networks much more stable.
"""

part5_q2 = r"""
**1. Comparison to Experiment 1.1:**
The general trend regarding depth remains consistent with Experiment 1.1: shallow networks ($L=2, 4$) train well, while the deeper network ($L=8$) completely fails to train (stuck at ~10% accuracy) due to the vanishing gradient problem.

**2. Effect of Width ($K$):**
* **For shallow depths ($L=2, 4$):** Increasing the width ($K$) leads to better accuracy. We observe that $K=128$ consistently outperforms $K=32$. This is expected because usually, wider networks have a larger capacity to represent complex functions, assuming they can be trained.
* **For deep depths ($L=8$):** Increasing the width had **no effect**. The network remained untrainable even with $K=128$. This demonstrates that the vanishing gradient problem is a structural issue related to depth; simply adding more parameters (width) cannot resolve the fact that the gradient signal is lost before reaching the early layers.
"""

part5_q3 = r"""
**1. Results Analysis:**
The experiment shows a limit to trainability. The shallowest network, $L=2$, trained successfully, reaching approximately 70% accuracy. However, increasing the depth slightly to $L=3$ and $L=4$ caused the model to completely fail, remaining at random guess accuracy (~10%).

**2. Impact of Pyramid Architecture:**
This failure at $L=3$ is notable because in Experiment 1.1, a fixed-width network of depth $L=4$ was easily trainable. This indicates that the "pyramid" structure (doubling filters $64 \to 128 \to 256$) introduces additional instability. Rapidly changing the number of channels between layers alters the signal variance significantly. Without Batch Normalization to compensate for these statistical shifts, the gradients vanish or explode much faster than in a fixed-width architecture, making even moderately shallow networks ($L=3$) impossible to train.
"""

part5_q4 = r"""
**1. Effect of Residual Connections:**
The results show that adding residual connections fixes the training problems we saw earlier. Unlike the plain CNNs, the ResNet models were able to train successfully even at deeper layers.

**2. Comparison to Experiment 1.1 (Fixed Width):**
In Exp 1.1, the network completely failed for depths $L=8$ and $L=16$ (stuck at 10% accuracy). Here, with the exact same configuration ($K=32$) but using ResNet, those depths trained well and reached high accuracy (~60-70%). Even the very deep $L=32$ network managed to learn something, which was impossible with the plain CNN.

**3. Comparison to Experiment 1.3 (Pyramid):**
In Exp 1.3, the "pyramid" structure ($K=64 \to 128 \to 256$) was unstable and failed even at shallow depths like $L=3$. With ResNet, this instability is gone—the model trained successfully for $L=4$ and $L=8$ with the same pyramid structure.

**Conclusion:**
The skip connections act as highways for the gradients, allowing them to propagate back to the early layers without vanishing. This solves the optimization issues that caused the plain CNNs to fail in the previous experiments.
"""

part5_q5 = r"""
"""

# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**1. Inference Analysis:**
* **Image 1 (Dolphins):** **Failure.** The model failed to identify the dolphins. Instead, it detected a "person" (0.90 confidence) and a "surfboard" (0.37).
* **Image 2 (Pets):** **Partial/Noisy Detection.** The model struggled with the overlapping animals. It misclassified the black dog as a "cat", and for the white dog, it outputted two conflicting boxes (one "dog", one "cat") on the same object.

**2. Reasons for Failure and Fixes:**
* **Unsupported Class (Dolphins):** The model maybe failed because Dolphin might **not be included in its training classes**. A neural network cannot predict a class it has no output neuron for. Instead, it matched the input features to the closest shapes it does know: a "surfboard" (smooth, in water) or a "person" (jumping action).
    * *Fix:* **Fine-Tuning.** We must retrain the final classification layer on a new dataset that explicitly contains the "Dolphin" class.
* **Occlusion/Overlap (Pets):** The dogs and cat are physically touching and overlapping. This causes two issues:
    1.  **Feature Mixing:** The convolutional features for the animals blend together, confusing the classifier (hence calling a dog a "cat").
    2.  **NMS Failure:** The Non-Maximum Suppression algorithm failed to remove the duplicate box on the white dog because the model gave them different labels (Cat vs Dog), so it treated them as two valid objects.
    * *Fix:* **Augmentation.** Retrain using "MixUp" or "Mosaic" augmentations (overlapping images) to force the model to handle occlusion better.

**3. Adversarial Attack on Object Detection:**
We can use a PGD attack, which is similar to what we learned for classification but with a different target.
* **Goal:** Instead of minimizing loss to *improve* accuracy, we want to **maximize** the loss to *destroy* detection.
* **Method:** We take the input image $x$ and add small noise. We perform **Gradient Ascent** on the "Objectness" score (the probability that a box contains an object).
* **Result:** By maximizing the loss of the objectness score, we force the model's confidence below the detection threshold. The bounding box will disappear, and the object will become "invisible" to the detector. Alternatively, we could attack the classification loss to force the model to misclassify the object (e.g., make a person look like a surfboard).
"""

part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**1. Camouflage (Moth): Failure.**
* **Result:** No objects were detected.
* **Analysis:** The moth's texture matches the wood grain background. YOLO relies heavily on **edge detection** and **contrast** to propose regions of interest. Since the camouflage effectively removes the boundary between the object and the background, the model treats the moth as part of the tree bark.

**2. Occlusion (Cat in Grass): Failure.**
* **Result:** No objects were detected.
* **Analysis:** The tall grass breaks the cat's body into disjointed fragments. Convolutional Neural Networks look for specific **spatial hierarchies of features** (e.g., ears near eyes near a nose). When these features are visually separated, as they are hidden by grass, the model cannot combine them into a single valid "Cat" instance, dropping the confidence score below the detection threshold.

**3. Motion Blur (Speeding Car): Misclassification.**
* **Result:** The main speeding car was misclassified as a **"boat" (0.42)**. However, the model correctly detected the tiny car in the distance as a **"car" (0.47)**.
* **Analysis:** This highlights the model's dependency on sharp, high-frequency features. The motion blur smoothed out defining characteristics of the main car like **wheels and door handles**, leaving a smooth, elongated shape that the model interpreted as a boat hull. The distant car was detected because it was less blurred, preserving the sharp edges required for identification despite its small size.
"""

part6_bonus = r"""
We attempted to improve detection using standard image processing techniques, but observed an interesting failure case:

1. **Sharpening (Motion Blur):** We used the detailEnhance function. Which boosts high-frequency details (like the wheels) that was smoothed out by the motion blur. we thought this would allow the model to recover the "Car" classification.
**Result** In reality, the model, which is sensitive to texture, likely got confused by the noise, preventing it from recognizing the underlying car geometry. **performance degraded**.

2. **Contrast (Moth):** We tried to boost contrast between the moth's wings and the wood, which we thought would make the object's boundaries distinct enough for the detector to propose a bounding box.
**Result:** While visually clearer to humans, the extreme contrast adjustment likely destroyed the subtle texture difference the model uses to identify biological forms, resulting in **no detection**.
 
3. **Occlusion (Cat):** We applied both **contrast enhancement and sharpening** to try and make the visible features (eyes, ears) to be noticed out from the grass.
**Result** No detection again, similar to both of the earlier attempts.
"""