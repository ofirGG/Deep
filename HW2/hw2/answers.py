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
    raise NotImplementedError()
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
    raise NotImplementedError()
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
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
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