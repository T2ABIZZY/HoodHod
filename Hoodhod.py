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

class metrices:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    @staticmethod
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

class BinaryDecisionTree:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return LeafNode(y)

        best_feature, best_value = self.get_best_split(X, y)
        left_mask = X[:, best_feature] < best_value
        right_mask = X[:, best_feature] >= best_value

        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return InternalNode(best_feature, best_value, left_tree, right_tree)

    def get_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_value = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            feature_values = sorted(set(X[:, feature]))
            for i in range(1, len(feature_values)):
                threshold = (feature_values[i - 1] + feature_values[i]) / 2
                left_mask = X[:, feature] < threshold
                right_mask = X[:, feature] >= threshold

                left_labels = y[left_mask]
                right_labels = y[right_mask]

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                gain = self.gain(y, left_labels, right_labels)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = threshold

        return best_feature, best_value

    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])

    @staticmethod
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def gain(self, parent_labels, left_labels, right_labels):
        parent_entropy = self.entropy(parent_labels)
        n = len(parent_labels)
        left_entropy = self.entropy(left_labels)
        right_entropy = self.entropy(right_labels)
        left_weight = len(left_labels) / n
        right_weight = len(right_labels) / n
        return parent_entropy - left_weight * left_entropy - right_weight * right_entropy


class InternalNode:
    def __init__(self, feature, value, left, right):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def predict(self, x):
        if x[self.feature] < self.value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class LeafNode:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, x):
        return self.labels[0]