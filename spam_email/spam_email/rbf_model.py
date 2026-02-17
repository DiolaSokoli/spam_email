import numpy as np

class RBF:
    def __init__(self, centers, sigma):
        self.centers = centers
        self.sigma = sigma

    def _gaussian(self, x, c):
        return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * self.sigma ** 2))

    def _design_matrix(self, X):
        G = np.zeros((X.shape[0], len(self.centers)))
        for i in range(X.shape[0]):
            for j in range(len(self.centers)):
                G[i, j] = self._gaussian(X[i], self.centers[j])
        return G

    def fit(self, X, y):
        G = self._design_matrix(X)
        self.weights = np.linalg.pinv(G) @ y

    def predict(self, X):
        G = self._design_matrix(X)
        y_hat = G @ self.weights
        return (y_hat >= 0.5).astype(int)
