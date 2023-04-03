import numpy as np

class LinearRegressionModel:
        def __init__(self, alpha=0.01, num_iterations=1000, threshold=0.5):
            self.alpha = alpha
            self.num_iterations = num_iterations
            self.threshold = threshold
            self.theta = None

        def fit(self, X, y):
            # Add a column of 1's to X for the bias term
            X = np.c_[np.ones((X.shape[0], 1)), X]
            # Initialize theta randomly
            self.theta = np.random.randn(X.shape[1])
            # Gradient descent
            for i in range(self.num_iterations):
                # Compute the predictions
                y_pred = X.dot(self.theta)
                # Compute the error
                error = y_pred - y
                # Update the parameters
                self.theta -= self.alpha * (1/X.shape[0]) * X.T.dot(error)

        def predict(self, X, binary=False):
            # Add a column of 1's to X for the bias term
            X = np.c_[np.ones((X.shape[0], 1)), X]
            # Compute the predictions
            y_pred = X.dot(self.theta)
            if binary:
                # Apply the threshold to convert predictions to binary decisions
                y_pred = (y_pred >= self.threshold).astype(int)
            return y_pred

        def score(self, X, y):
            # Add a column of 1's to X for the bias term
            X = np.c_[np.ones((X.shape[0], 1)), X]
            # Compute the predictions
            y_pred = X.dot(self.theta)
            # Compute the R^2 score
            r2 = 1 - ((y - y_pred)**2).sum() / ((y - y.mean())**2).sum()
            return r2

        def mean_squared_error(self, X, y):
            y_pred = self.predict(X)
            return np.mean((y - y_pred) ** 2)

        def r_squared(self, X, y):
            y_pred = self.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot

        def mean_absolute_error(self, X, y):
            y_pred = self.predict(X)
            return np.mean(np.abs(y_pred - y))