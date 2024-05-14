from abc import ABC, abstractmethod

class BaseMLModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def train(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model.
        """
        pass
