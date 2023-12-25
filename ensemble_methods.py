import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, n_estimators=100, max_depth=None, threshold=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.threshold = threshold
        self.models = [DecisionTreeClassifier(max_depth=self.max_depth) for _ in range(n_estimators)]

    def fit(self, x, y):
        for model in self.models:
            indices = np.random.choice(len(x), len(x), replace=True)
            x_subset, y_subset = x.iloc[indices], y.iloc[indices]
            model.fit(x_subset, y_subset)

        return self

    def predict(self, x):
        pred = np.zeros((len(x), self.n_estimators))
        for i, model in enumerate(self.models):
            pred[:, i] = model.predict(x)
        avg_predictions = np.mean(pred, axis=1)
        binary_predictions = (avg_predictions >= self.threshold).astype(int)
        return binary_predictions

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'threshold': self.threshold,
        }

    def set_params(self, **params):
        if not params:
            return self

        for param, value in params.items():
            setattr(self, param, value)

        return self


class AdaBoost:
    def __init__(self, n_estimators=50, max_depth=4):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        self.max_depth = max_depth

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        # Get the number of rows and columns
        n_samples, n_features = x.shape
        # Initialize the weights to 1/N
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(x, y, sample_weight=weights)
            y_pred = model.predict(x)
            weighted_error = np.sum(weights[y_pred != y])
            if weighted_error >= 0.5:
                break

            alpha = 0.5 * np.log((1.0 - weighted_error) / max(weighted_error, 1e-10))
            self.alphas.append(alpha)
            self.models.append(model)

            # Update weights
            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)

        return self

    def predict(self, x):
        x = np.array(x)
        pred = np.zeros(len(x))
        for alpha, model in zip(self.alphas, self.models):
            pred += alpha * model.predict(x)
        return np.sign(pred)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
        }

    def set_params(self, **params):
        if not params:
            return self

        for param, value in params.items():
            setattr(self, param, value)

        return self


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.models = []

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        n_samples, n_features = x.shape
        for _ in range(self.n_estimators):
            # Randomly select a subset of features
            selected_features = np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False)
            x_subset = x[:, selected_features]

            # Create a decision tree with random features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(x_subset, y)
            self.models.append((tree, selected_features))
        return self

    def predict(self, x):
        x = np.array(x)
        pred = np.zeros((x.shape[0], self.n_estimators))
        for i, (tree, selected_features) in enumerate(self.models):
            x_subset = x[:, selected_features]
            pred[:, i] = tree.predict(x_subset)

        # Use majority voting for the final prediction
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=pred)
        return final_predictions

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }

    def set_params(self, **params):
        if not params:
            return self

        for param, value in params.items():
            setattr(self, param, value)

        return self
