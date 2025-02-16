import numpy as np

class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:
        """
        Args:
            X: (N,D) numpy array corresponding to a dataset

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        centered = X - np.mean(X, axis=0)
        self.U, self.S, self.V = np.linalg.svd(centered, full_matrices=False)

    def transform(self, data: np.ndarray, K: int=2) -> np.ndarray:
        """
        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        self.fit(data)
        pcs = self.U * self.S
        return pcs[:,:K]

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99) -> np.ndarray:
        """
        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        self.fit(data)
        K = np.argmax(np.cumsum(np.square(self.S)) / np.sum(np.square(self.S)) >= retained_variance) + 1
        K = max(K, 2) # min 2 features after PCA
        return self.transform(data, K)