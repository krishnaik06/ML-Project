import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def ml_project_lifecycle():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow experiment
    mlflow.start_run()

    # Set experiment parameters
    params = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predict and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # End MLflow experiment
    mlflow.end_run()

if __name__ == "__main__":
    ml_project_lifecycle()
