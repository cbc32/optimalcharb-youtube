import numpy as np
from typing import Callable

def root_mean_squared_error(y_truth, y_pred):
    return np.sqrt(np.mean((y_truth - y_pred) ** 2))

def mean_absolute_error(y_truth, y_pred):
    return np.mean(np.abs(y_truth - y_pred))

def accuracy(y_truth, y_pred):
    return np.mean(np.equal(y_truth, y_pred))

class Regression:
    def __init__(self):
        self.num_features = None

    def add_bias(self, X: np.array):
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def check_num_features(self, X: np.array):
        if X.shape[1] != self.num_features:
            raise ValueError("Number of features in X must match number of features in training data")

    def fit(self, X: np.array, y: np.array):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X must match number of samples in y")
        self.num_features = X.shape[1]

    def grad_descent(self, beta0: np.array, grad: Callable[[np.array], np.array], learning_rate: float, epochs: int) -> np.array:
        beta = beta0.copy()
        for _ in range(epochs):
            beta -= learning_rate * grad(beta)
        return beta

class LinearRegression(Regression):
    def __init__(self):
        super().__init__()
        self.beta = None

    def predict(self, X: np.array) -> np.array:
        self.check_num_features(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coefficients

    def fit(self, X: np.array, y: np.array):
        super().fit(X, y)
        X_b = self.add_bias(X)
        self.beta = np.linalg.pinv(X_b) @ y

    def grad(self, beta: np.array, X_b: np.array, y: np.array) -> np.array:
        return -2 / X_b.shape[0] * X_b.T @ (y - X_b @ beta)

    def fit_gd(self, X: np.array, y: np.array, learning_rate=0.01, epochs=10000):
        super().fit(X, y)
        X_b = self.add_bias(X)
        self.beta = self.grad_descent(np.zeros(X_b.shape[1]), lambda beta: self.grad(beta, X_b, y), learning_rate, epochs)

    def fit_sgd(self, X: np.array, y: np.array, learning_rate=0.01, epochs=10000):
        super().fit(X, y)
        X_b = self.add_bias(X)
        for _ in range(epochs):
            i = np.random.randint(X_b.shape[0])
            self.beta -= learning_rate * self.grad(self.beta, X_b[i:i+1], y[i])

class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def fit(self, X: np.array, y: np.array):
        super(LinearRegression, self).fit(X, y)
        X_b = self.add_bias(X)
        self.beta = np.linalg.pinv(X_b.T @ X_b + self.alpha * np.eye(X_b.shape[1])) @ X_b.T @ y

    def grad(self, beta: np.array, X_b: np.array, y: np.array) -> np.array:
        return -2 * X_b.T @ (y - X_b @ beta) + 2 * self.alpha * beta 
    
    def fit_sgd(self, X: np.array, y: np.array, learning_rate=0.01, epochs=1000):
        # because the grad is different between gd and sgd, we need to override the fit_sgd method
        super(LinearRegression, self).fit(X, y)
        X_b = self.add_bias(X)
        self.beta = np.zeros(X_b.shape[1])
        for _ in range(epochs):
            i = np.random.randint(X_b.shape[0])
            grad = -2 * X_b[i:i+1].T @ (y[i] - X_b[i:i+1] @ self.beta) + 2 * self.alpha * self.beta / X_b.shape[0]
            self.beta -= learning_rate * grad

class LassoRegression(LinearRegression):
    def __init__(self, alpha=1.0, tol=1e-4, max_iter=1000):
        super().__init__()
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def soft_thresholding(self, rho, alpha):
        if rho < -alpha:
            return rho + alpha
        elif rho > alpha:
            return rho - alpha
        else:
            return 0.0

    def fit(self, X: np.array, y: np.array):
        super().fit(X, y)
        X_b = self.add_bias(X)
        self.beta = np.zeros(X_b.shape[1])
        for _ in range(self.max_iter):
            beta_old = self.beta.copy()
            for j in range(X_b.shape[1]):
                X_j = X_b[:, j]
                residual = y - X_b @ self.beta + self.beta[j] * X_j
                rho = X_j.T @ residual
                if j == 0:  # no regularization on intercept term
                    self.beta[j] = rho / X_b.shape[0]
                else:
                    self.beta[j] = self.soft_thresholding(rho, self.alpha) / (X_j.T @ X_j)
            if np.sum(np.abs(self.beta - beta_old)) < self.tol:
                break

    def grad(self, beta: np.array, X_b: np.array, y: np.array) -> np.array:
        raise NotImplementedError("Lasso regression does not have a closed form gradient")

class Classification(Regression):
    def __init__(self):
        super().__init__()
        self.classes = None
        self.num_classes = None

    def encode(self, y: np.array):
        return np.array([np.where(self.classes == label)[0][0] for label in y])
    
    def one_hot_encode(self, y: np.array):
        return np.eye(self.num_classes)[y]
    
    def fit(self, X, y):
        super().fit(X, y)
        self.classes = np.unique(y)
        self.num_classes = self.classes.shape[0]

class LogisticRegression(Classification):
    def __init__(self):
        super().__init__()
        self.beta = None
        self.loss_history = []
        self.accuracy_history = []

    def loss(self, probs: np.array, y: np.array):
        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    def softmax(self, z: np.array):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict_probs(self, X: np.array):
        self.check_num_features(X)
        X_b = self.add_bias(X)
        return self.softmax(X_b @ self.beta)
    
    def predict(self, X: np.array):
        return self.classes[np.argmax(self.predict_probs(X), axis=1)]
    
    def fit(self, X: np.array, y: np.array, learning_rate=1, epochs=1000, sgd=False):
        super().fit(X, y)
        X_b = self.add_bias(X)
        y_enc = self.one_hot_encode(self.encode(y))
        self.beta = np.zeros((self.num_features + 1, self.num_classes))
        self.loss_history = self.accuracy_history = []
        for _ in range(epochs):
            probs = self.softmax(X_b @ self.beta)
            grad = -X_b.T @ (y_enc - probs) / X_b.shape[0]
            self.beta -= learning_rate * grad
            self.loss_history.append(self.loss(probs, y_enc))
            self.accuracy_history.append(accuracy(y, self.predict(X)))

class NaiveBayes(Classification):
    def __init__(self):
        super().__init__()
        self.class_priors = None
        self.feature_means = None
        self.feature_variances = None

    def fit(self, X: np.array, y: np.array):
        super().fit(X, y)
        y = self.encode(y)
        self.class_priors = np.zeros(self.num_classes)
        self.feature_means = np.zeros((self.num_classes, self.num_features))
        self.feature_variances = np.zeros((self.num_classes, self.num_features))
        for c in range(self.num_classes):
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            self.feature_means[c] = np.mean(X_c, axis=0)
            self.feature_variances[c] = np.var(X_c, axis=0)
    
    def predict_logprobs(self, X: np.array):
        self.check_num_features(X)
        epsilon = 1e-10
        log_probs = np.array([[np.log(self.class_priors[c] + epsilon) for c in range(self.num_classes)] for _ in range(X.shape[0])])
        for c in range(self.num_classes):
            mean = self.feature_means[c]
            var = self.feature_variances[c] + epsilon
            log_probs[:,c] += -0.5 * np.sum(np.log(2 * np.pi * var))
            log_probs[:,c] += -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        return log_probs

    def predict_probs(self, X: np.array):
        log_probs = self.predict_logprobs(X)
        return np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))

    def predict(self, X: np.array):
        return self.classes[np.argmax(self.predict_logprobs(X), axis=1)]
    
class Perceptron(Classification):
    def __init__(self):
        super().__init__()
        self.beta = None
        self.loss_history = []
        self.accuracy_history = []
    
    def loss(self, y_pred: np.array, y_truth: np.array):
        return np.mean(y_pred != y_truth)
    
    def predict(self, X: np.array):
        self.check_num_features(X)
        X_b = self.add_bias(X)
        return self.classes[np.argmax(X_b @ self.beta, axis=1)]
    
    def fit(self, X: np.array, y: np.array, learning_rate=1, epochs=1000):
        super().fit(X, y)
        y_idx = self.encode(y)
        X_b = self.add_bias(X)
        self.beta = np.zeros((self.num_features + 1, self.num_classes))
        self.loss_history = self.accuracy_history = []
        for _ in range(epochs):
            for i in range(X_b.shape[0]):
                y_pred = np.argmax(X_b[i] @ self.beta)
                if y_pred != y_idx[i]:
                    self.beta[:, y_idx[i]] += learning_rate * X_b[i]
                    self.beta[:, y_pred] -= learning_rate * X_b[i]
            self.loss_history.append(self.loss(self.predict(X), y))
            self.accuracy_history.append(accuracy(y, self.predict(X)))

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LinearRegression()
    model.fit_gd(X, y)
    print(model.beta)
    model.fit(X, y)
    print(model.beta)
    model.fit_sgd(X, y)
    print(model.beta)

    model1 = RidgeRegression()
    model1.fit_gd(X, y)
    print(model1.beta)
    model1.fit_sgd(X, y)
    print(model1.beta)
    model1.fit(X, y)
    print(model1.beta)

    model2 = LassoRegression()
    model2.fit(X, y)
    print(model1.beta)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array(["cat", "dog", "bird", "bird"])

    model3 = LogisticRegression()
    model3.fit(X, y)
    print(model3.beta)
    print(model3.predict(X))

    model4 = NaiveBayes()
    model4.fit(X, y)
    print(model4.feature_means)
    print(model4.feature_variances)
    print(model4.class_priors)
    print(model4.predict(X))

    model5 = Perceptron()
    model5.fit(X, y)
    print(model5.beta)
    print(model5.predict(X))