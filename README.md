# First Lab KI-Systeme

Over all task is to build a robot that can detect litter and notify its operator.

This project was build with the autoresaerch idea of Andrew Karpathy: https://github.com/karpathy/autoresearch

The overall idea is to critically look at the experiments and progress the AI made, identify improvements and integrate a further improved version into a robot setup.

Other approaches fine-tune a yolo model: e.g. see for https://github.com/jeremy-rico/litter-detection

## 1 Student Task

- [Task Description](docs/student_task.md)
- [Context to this project](docs/explainer.md)

## 2 Student Task

- [Task Description 2](docs/student_task_2.md)

## Example images not in the dataset

|No litter | Litter |
|---|---|
|![](docs/images/Image2.jpeg) | ![](docs/images/Image3.jpeg) |

## Autoresearch Content

> Note: There is already one good model in this repository. Thus you should be able to investigate the performance using the Analysis Notebook.

- [Analysis Notebook](auto-research/analysis.ipynb)
- [Instructions](auto-research/program.md)
- [Finding from previous runs](auto-research/findings.md)

## Setup

Init project:

```bash
uv sync
```

Content:

- There is an [analysis notebook](auto-research/analysis.ipynb) to take a first look on the project and test the existing models.
- The project contains a mlflow project that stores the hole experiment and training history.
  Run the following command to launch the mlflow server and ui
  ```bash
  uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --default-artifact-root ./artifacts/mlflow/mlruns
  ```





## Additional Content

- [Experiment Tracking](https://mlflow.org/docs/latest/ml/getting-started/deep-learning/)
