r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 64
    hypers['h_dim'] = 256
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.8
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

1. **Memory Constraints (VRAM):** Training on the whole text at once would require unrolling the computation graph for the entire length of the corpus. This would consume an impossible amount of memory for storing gradients during Backpropagation Through Time (BPTT). Splitting into sequences allows for Truncated BPTT, which fits in GPU memory.

2. **Vanishing/Exploding Gradients:** In extremely long sequences, gradients propagated backward over thousands of time steps tend to either vanish to zero (stopping learning) or explode to infinity (instability). Limiting the sequence length mitigates this issue.

3. **Optimization Efficiency:** Splitting the text creates many "samples," allowing us to use mini-batch Stochastic Gradient Descent. This updates the weights frequently (thousands of times per epoch) rather than once per epoch, leading to much faster convergence.

4. **Data Parallelism:** Small sequences allow us to stack data into batches (e.g., 64 sequences at a time), maximizing the parallel processing power of the GPU.
"""

part1_q2 = r"""
**Your answer:**

It is possible because we maintain and propagate the hidden state across batches.

1.  **Hidden State Persistence:** Although we perform "Truncated Backpropagation Through Time" (cutting off the gradients after seq_len steps to save memory), we do not reset the values of the hidden state between batches. The final hidden state of batch $i$ is fed as the initial hidden state for batch $i+1$.

2.  **Forward vs. Backward Horizon:** While the gradients (learning signal) can only flow back seq_len seps, the forward pass (context information) flows indefinitely. This allows the network to maintain a "memory" of events that happened many batches ago.

3.  **Contiguous Sampling:** Our SequenceBatchSampler ensures that the batches are ordered sequentially (the start of batch $i+1$ is the continuation of batch $i$). This alignment allows the persistent hidden state to carry valid context from the previous segment of text to the current one.
"""

part1_q3 = r"""
**Your answer:**

We do not shuffle because we need to preserve contextual continuity between batches to simulate a much longer sequence.

1.  **Stateful Training:** We are training in a "stateful" manner, where the final hidden state of batch $i$ is used as the initial hidden state for batch $i+1$. This allows the model's memory (hidden state) to persist over time.

2.  **Logical Flow:** If we shuffled the batches, the hidden state passed to the network would be from a completely unrelated part of the text (e.g., passing the context of "Chapter 10" into the start of "Chapter 1"). This would break the narrative flow and force the model to reset its context constantly, limiting its effective memory to just the seq_len (e.g., 64 characters).

3.  **Long-Term Dependencies:** By keeping the order sequential, the forward pass can carry information across thousands of characters (through the hidden state), even though the backward pass (gradient calculation) is truncated to only 64 steps.
"""

part1_q4 = r"""
**Your answer:**

1. **Why lower the temperature (< 1.0)?**
   We lower the temperature to make the model more confident and conservative. A temperature $T < 1$ sharpens the probability distribution, exaggerating the differences between high and low probability characters. This filters out "noise" (low probability options) and ensures the generated text is more coherent, grammatical, and structurally correct, as the model sticks to the patterns it learned most strongly.

2. **What happens when T is very high (> 1.0)?**
   The distribution becomes flatter (more uniform). The differences between likely and unlikely characters shrink.
   Result: The model takes more risks and selects low-probability characters more often. The text becomes chaotic, contains spelling errors, and eventually turns into gibberish because the model is ignoring its own learned probabilities.

3. **What happens when T is very low (close to 0)?**
   The distribution becomes extremely sharp (approximating an argmax). The probability of the most likely character approaches 1, while all others approach 0.
   Result: The model becomes deterministic and repetitive. It will often get stuck in loops (e.g., "and the and the and the") because it constantly picks the single most "safe" next character without any variation to break the cycle.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 128
    hypers['z_dim'] = 128
    hypers['x_sigma2'] = 0.9
    hypers['learn_rate'] = 2e-4
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The hyperparameter $\sigma^2$ represents the assumed variance of the data likelihood $p(\mathbf{x}|\mathbf{z}) \sim \mathcal{N}(\Psi(\mathbf{z}), \sigma^2 \mathbf{I})$. In the loss function, it acts as a weighting coefficient that controls the trade-off between the reconstruction loss and the KL-divergence regularization.

1.  **Low $\sigma^2$:**
    Effect: The reconstruction error term is scaled up (large weight). The model prioritizes minimizing pixel-wise differences between the input and output.
    Result: The model produces sharp, high-quality reconstructions. However, it may ignore the KL-divergence term, leading to a latent space that is not smooth or normally distributed (overfitting). This results in poor random sampling (generation) quality.

2.  **High $\sigma^2$:**
    Effect: The reconstruction error term is scaled down (small weight). The KL-divergence term becomes relatively more important.
    Result: The model prioritizes fitting the latent distribution to the prior $\mathcal{N}(0, I)$. The latent space becomes very regularized and smooth. However, the model may fail to encode enough information about the input, leading to blurry reconstructions and generic features (a phenomenon known as "posterior collapse").
"""

part2_q2 = r"""
**Your answer:**

1.  **Purpose of the Loss Terms:**
    Reconstruction Loss: This term measures how well the decoded image matches the original input. Its purpose is to force the latent representation ($z$) to capture the meaningful "content" and information required to reproduce the data. Without this, the model would produce random noise.
    KL Divergence Loss: This term measures the difference (divergence) between the predicted latent distribution $q(\mathbf{z}|\mathbf{x})$ and a fixed standard normal prior $p(\mathbf{z}) = \mathcal{N}(0, I)$. Its purpose is to regularize the latent space structure.

2.  **Effect on Latent-Space Distribution:**
    The KL loss forces the encoder to map inputs to a compact, continuous, and overlapping region centered around the origin. It prevents the encoder from memorizing data points as isolated, far-apart "dots" (delta functions) or cheating by setting the variance to zero. Instead, it forces every input to be mapped to a probability cloud that resembles a standard Gaussian.

3.  **Benefit of this Effect:**
    This regularization is crucial for the VAE's generative capability:
    Sampling: Because we forced the training distribution to look like a standard normal distribution, we know exactly how to generate new data: simply sample $z \sim \mathcal{N}(0, I)$ and decode it.
    Smoothness (Interpolation): The compact space ensures that there are no "gaps" or "holes" in the latent space. Walking from one point to another in the latent space results in a smooth semantic transition in the image space, rather than sudden jumps or garbage outputs.
"""

part2_q3 = r"""
**Your answer:**

We start by maximizing the evidence distribution $p(\mathbf{X})$ because Maximum Likelihood Estimation (MLE) is the fundamental objective of generative modeling.

1.  **The Goal:** Our goal is to train a model that learns the true underlying probability distribution of the data, $p_{data}(\mathbf{x})$. If our model assigns a high probability $p_\theta(\mathbf{x})$ to the real samples we observed, it means the model has successfully learned to recognize and generate data that looks like the training set.

2.  **The Problem:** Direct maximization of $p(\mathbf{x})$ is intractable. In a latent variable model, calculating the likelihood of a single image requires marginalizing over all possible latent variables $z$:
    $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z}) d\mathbf{z}$$
    This integral is impossible to compute for complex neural networks.

3.  **The Solution (ELBO):** Since we cannot maximize $\log p(\mathbf{x})$ directly, we derive the Evidence Lower Bound (ELBO). By maximizing the ELBO, we implicitly maximize the log-likelihood of the data (or at least push its lower bound up), effectively training the model to satisfy the MLE objective.
"""

part2_q4 = r"""
**Your answer:**

We model the log-variance ($\log \sigma^2$) instead of the variance directly for two main reasons: numerical stability and mathematical constraints.

1.  **Range Constraint (Positivity):** Variance ($\sigma^2$) is mathematically required to be a non-negative number ($[0, \infty)$). Standard neural network layers (like Linear layers) produce outputs in the range $(-\infty, \infty)$.
    If we predicted $\sigma^2$ directly, we would need to enforce positivity using an activation function like ReLU or Softplus. ReLU could lead to "dead neurons" (variance = 0), causing infinite densities and numerical crashes.
    By predicting $x = \log \sigma^2$, we can allow the network to output any real number. We then compute $\sigma^2 = e^x$, which is guaranteed to be strictly positive.

2.  **Numerical Stability & Sensitivity:**
    It effectively allows the network to learn the scale of the variance.
    Small changes in the log-space correspond to multiplicative changes in the variance space. This makes it easier for the optimizer to fine-tune very small variances (which map to negative numbers like -5, -10) without dealing with tiny floating-point numbers directly during backpropagation.
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        window_size=16,
        droupout=0.1,
        lr=0.0001,
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

Stacking encoder layers with sliding-window attention increases the **receptive field** of each token, similar to how stacking layers in a Convolutional Neural Network (CNN) increases the area of the input image that a deep filter can "see."

1. **Local View in Initial Layers:** In the first layer, each token can only attend to its immediate neighbors within the fixed window size $w$. For a query at position $i$, the context is limited to $[i - w/2, i + w/2]$.

2. **Context Propagation through Depth:** In the second layer, each token's representation already contains information aggregated from its own window in the first layer. Therefore, when a query at position $i$ attends to a neighbor at position $i + w/2$, it is implicitly receiving information that the neighbor gathered from *its* neighbors further down the sequence (up to $i + w$).

3. **Linear Receptive Field Growth:** With each additional layer $l$, the effective context window (receptive field) expands. For a transformer with $L$ layers and a window size $w$, the top-layer representation of a token can "see" a total context of approximately $L \times w$ tokens.

4. **Global Reach with Fewer Parameters:** This mechanism allows the model to build a global understanding of the text by propagating information across layers, even though each individual attention operation remains computationally efficient at $O(n \times w)$ instead of $O(n^2)$.
"""

part3_q2 = r"""
**Your answer:**

**Proposed Variation: Dilated Sliding Window Attention**

Instead of attending to the $w$ immediate neighbors ($i \pm 1, i \pm 2, \dots$), we attend to neighbors with a "gap" (dilation) of size $d$. For example, with $d=2$, a token at position $i$ attends to indices $i \pm 2, i \pm 4, \dots$.

1.  **Computational Complexity:**
    The complexity remains **$O(n \cdot w)$**.
    Even though the attention window spans a larger distance in the text, the *number* of attention scores computed per token is still fixed at $w$. We simply skip over the tokens in the gaps.

2.  **Global Information Sharing:**
    Global information is shared much faster because the receptive field expands more rapidly.
    * In standard sliding window, the receptive field grows linearly ($L \times w$).
    * In dilated attention, the receptive field is $L \times w \times d$.
    If we increase the dilation rate exponentially with each layer (e.g., Layer 1 has $d=1$, Layer 2 has $d=2$, Layer 3 has $d=4$), the receptive field grows exponentially. This allows the model to "see" the entire sequence with very few layers (logarithmic depth).

3.  **Limitations:**
    * **The "Gap" Problem:** By skipping tokens, the model loses detailed local information. A token might attend to a word 10 positions away but miss the word immediately next to it.
    * **Solution:** This is typically mitigated by using a "Hybrid" approach: some heads (or layers) use standard sliding window (for local details) while others use dilated window (for global context).
"""

# ==============
