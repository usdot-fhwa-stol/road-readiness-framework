# Lane Eval Adapters

A task-aware evaluation framework for lane detection across multiple datasets and models.

Initial scope:

- Models: YOLOPX
- Datasets: BDD100K, TuSimple, CULane, CurveLanes
- Task: lane detection / lane segmentation evaluation

Core idea:

```text
dataset adapter -> canonical lane target
model adapter   -> canonical lane prediction
evaluator       -> metrics

