# Paper Ideas

Objective of the paper: Inspired on real life challenges that are raised
when trying to track people:
a. Lack of training data.
b. Ability to switch from cold start to trained model at optimal time
c. Inference costs reduced by reducing the frame rate for tracking and
making use of flow model for prediction.
It introduces new evaluation metrics since previous evaluation metrics
for the task of mot are not comparable or useful to evaluate the newly
defined problem.
It introduces new evaluation dataset.
It introduces new paradigm and a sample algorithm.
It compares new paradigm with previous supervised trained or
non-trained alternatives.

1. Describe algorithm implementation

2. For both the linear model and the assignment model, 
explain and theoretically analyze the strategy to switch from
the algorithmic cold start to the trained model.

3. Evaluate how flow model is better than linear position prediction
and how assignment model is better than bipartite matching.

4. Evaluate how algorithm improves over time (and the improvement
it brings to the situation of lack of training data)

5. Evaluate how flow model can be trained at higher frame rate
and used at inference time at lower frame rate (evaluate how lowering
the frame rate degrades the performance of the algorithm).