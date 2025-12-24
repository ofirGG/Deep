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
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
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
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
