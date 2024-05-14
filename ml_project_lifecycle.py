import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BaseMLModel:
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

class MLProjectLifecycle(BaseMLModel):
    def __init__(self):
        self.params = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
        self.model = None

    def train(self):
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)

        return X_test, y_test

    def evaluate(self, X_test, y_test):
        # Predict and calculate accuracy
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def run(self):
        # Start MLflow experiment
        mlflow.start_run()

        # Log experiment parameters
        mlflow.log_params(self.params)

        # Train model and get test data
        X_test, y_test = self.train()

        # Evaluate model
        accuracy = self.evaluate(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(self.model, "model")

        # End MLflow experiment
        mlflow.end_run()

if __name__ == "__main__":
    project = MLProjectLifecycle()
    project.run()
