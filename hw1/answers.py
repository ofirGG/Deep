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
# Part 3 answers

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

# ==============
