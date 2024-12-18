import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.01, epochs=1000):
        self.alpha = alpha      # Learning rate
        self.epochs = epochs    # Number of iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        self.bias = 0                        # Initialize bias to zero

        # Gradient Descent
        for _ in range(self.epochs):
            # Linear Prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
