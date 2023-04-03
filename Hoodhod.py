import numpy as np
class Linear_Model:

    class LinearRegressionModel:
            def __init__(self, alpha=0.01, num_iterations=1000, threshold=0.5):
                self.alpha = alpha
                self.num_iterations = num_iterations
                self.threshold = threshold
                self.theta = None
            def fit(self, X, y):
                X = np.c_[np.ones((X.shape[0], 1)), X]
                self.theta = np.random.randn(X.shape[1])
                for i in range(self.num_iterations):
                    y_pred = X.dot(self.theta)
                    error = y_pred - y
                    self.theta -= self.alpha * (1/X.shape[0]) * X.T.dot(error)

            def predict(self, X, binary=False):
                X = np.c_[np.ones((X.shape[0], 1)), X]
                y_pred = X.dot(self.theta)
                if binary:
                    y_pred = (y_pred >= self.threshold).astype(int)
                return y_pred
            def score(self, X, y):
                X = np.c_[np.ones((X.shape[0], 1)), X]
                y_pred = X.dot(self.theta)
                r2 = 1 - ((y - y_pred)**2).sum() / ((y - y.mean())**2).sum()
                return r2
