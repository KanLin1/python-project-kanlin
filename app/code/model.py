import time
import numpy as np


class LogisticRegression:

    def __init__(
        self,
        k,
        n,
        method="batch",
        alpha=0.05,
        max_iter=500,
        penalty=None,
        l=0.0,
        random_state=42
    ):
        self.k = k
        self.n = n
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.penalty = penalty
        self.l = l
        self.random_state = random_state

    def fit(self, X, Y):
        rng = np.random.default_rng(self.random_state)

        self.W = rng.normal(
            0,
            0.01,
            size=(self.n, self.k)
        )

        self.losses = []
        start_time = time.time()

        for i in range(self.max_iter):

            if self.method == "batch":
                X_batch = X
                Y_batch = Y

            elif self.method == "minibatch":
                batch_size = int(0.3 * X.shape[0])

                idx = rng.choice(
                    X.shape[0],
                    size=batch_size,
                    replace=False
                )

                X_batch = X[idx]
                Y_batch = Y[idx]

            elif self.method == "sto":
                idx = rng.integers(
                    0,
                    X.shape[0]
                )

                X_batch = X[idx:idx + 1]
                Y_batch = Y[idx:idx + 1]

            else:
                raise ValueError(
                    'method must be "batch", '
                    '"minibatch" or "sto"'
                )

            loss, grad = self.gradient(
                X_batch,
                Y_batch
            )

            self.losses.append(loss)

            self.W = (
                self.W -
                self.alpha * grad
            )

        self.training_time_ = (
            time.time() -
            start_time
        )

        return self

    def gradient(self, X, Y):
        m = X.shape[0]

        h = self.h_theta(
            X,
            self.W
        )

        loss = -np.sum(
            Y * np.log(
                h + 1e-15
            )
        ) / m

        error = h - Y

        grad = (
            X.T @ error
        ) / m

        if self.penalty == "l2":
            W_reg = self.W.copy()
            W_reg[0, :] = 0.0

            loss += (
                self.l *
                np.sum(
                    W_reg ** 2
                )
            )

            grad += (
                2.0 *
                self.l *
                W_reg
            )

        return loss, grad

    def softmax(self, z):
        z = z - np.max(
            z,
            axis=1,
            keepdims=True
        )

        exp_scores = np.exp(z)

        return exp_scores / np.sum(
            exp_scores,
            axis=1,
            keepdims=True
        )

    def h_theta(self, X, W):
        return self.softmax(
            X @ W
        )

    def predict(self, X):
        return np.argmax(
            self.h_theta(
                X,
                self.W
            ),
            axis=1
        )
