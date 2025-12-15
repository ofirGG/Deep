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

**Yes, the second-order derivative can be very helpful.**

While standard gradient descent relies only on the first derivative (gradient) to determine the *direction* of the step, the second derivative (Hessian matrix) provides information about the **curvature** of the loss landscape.

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
    * **Behavior:** This model exhibits classic **overfitting**. It achieves very high training accuracy (>80%) and very low training loss, indicating it has memorized the training data.
    * **Evidence:** In the `test_loss` graph, the blue line starts low but then **increases dramatically** as training progresses. This divergence between the decreasing training loss and increasing test loss is the hallmark of overfitting—the model is becoming less generalizable as it continues to train.

* **With Dropout (`dropout=0.4` - Orange line):**
    * **Behavior:** The dropout acts as a regularizer, making it harder for the model to memorize the training data.
    * **Evidence:** The training accuracy is significantly lower (~45%) compared to the no-dropout model, but the **test loss is stable** and does not explode like the blue line. Crucially, the `test_acc` (bottom right) for the orange line eventually surpasses the blue line (reaching ~28% vs ~25%), showing that the model generalizes better to unseen data despite performing "worse" on the training set.

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
* **Accuracy** is a **discrete** metric: It only cares about which class has the highest score. It does not matter if the correct class wins by 0.01 or by 0.99.
* **Cross-Entropy Loss** is a **continuous** metric: It cares about the **confidence** (probability) assigned to the correct class.

**How it happens:**
Imagine a scenario where the model is evaluated on a test set:
1.  **Improving Confidence on Easy Samples:** For the majority of samples that the model is *already* classifying correctly, the model becomes much more confident (e.g., the probability of the correct class rises from 0.6 to 0.99). This causes a **massive drop** in the total cross-entropy loss.
2.  **failing Borderline Samples:** Simultaneously, a few "borderline" samples that were previously barely correct (e.g., probability 0.51) shift slightly to become barely incorrect (e.g., probability 0.49). This causes the **accuracy to drop**.

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
The error occurred because you tried to do **one backward pass on the sum of the losses**.
Most deep learning frameworks (like PyTorch) build a **computational graph** to track operations for backpropagation. By feeding batch after batch and summing the losses *without* detaching or backwarding, the framework keeps extending this massive graph in memory, storing all intermediate activations for *all* batches to eventually compute gradients. This effectively reconstructs the memory footprint of the full dataset, defeating the purpose of splitting it.

**3.3. Solution**
Instead of summing the *losses* and doing one backward pass, you should **accumulate the gradients**.
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
    raise NotImplementedError()
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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""