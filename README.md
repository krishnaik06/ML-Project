# ML-Project with MLflow Integration and Modular Coding

This repository demonstrates an end-to-end Machine Learning project lifecycle managed with MLflow, showcasing modular coding practices with classes and inheritance. It includes the integration of MLflow for experiment tracking, model management, and deployment, and organizes the project into a structured folder system for better maintainability and scalability.

## Project Structure

The project is organized into the following structured folders:

- `data/`: Contains datasets used in the project.
- `models/`: Contains the machine learning models, including base and derived classes.
- `scripts/`: Contains scripts for running experiments, training models, and evaluating performance.

## Setting Up the Project with MLflow

To set up this project with MLflow, ensure you have MLflow installed. You can install MLflow using pip:

```
pip install mlflow
```

Additionally, ensure all project dependencies are installed by running:

```
pip install -r requirements.txt
```

## Running Experiments and Managing Models with MLflow

To run experiments and manage models with MLflow, execute the script within the `scripts/` directory:

```
python scripts/ml_project_lifecycle.py
```

This script demonstrates the lifecycle of a Machine Learning project, including initializing an MLflow experiment, logging parameters, metrics, and models, and tracking experiments using modular coding practices.

## Further Reading

For more information on MLflow and its capabilities, refer to the [MLflow documentation](https://www.mlflow.org/docs/latest/index.html).
