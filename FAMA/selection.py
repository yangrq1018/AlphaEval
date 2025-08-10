import numpy as np
from sklearn.cluster import KMeans


class CrossSampleSelection():
    """Cross-sample selection for factor analysis."""
    
    def __init__(self, k: int) -> None:
        """
        Initialize the CrossSampleSelection with a specified number of clustering centers.

        Parameters
        ----------
        k : int
            Number of clustering centers.
        """
        self.k = k
    
    def fit(self, X: np.ndarray) -> KMeans:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        """
        print(f"Fitting KMeans with {self.k} clusters on data of shape {X.shape}")
        print("Unique rows:", np.unique(X, axis=0).shape[0])
        self.kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X)
        
        return self.kmeans
     
    def sample_select(self, l: int) -> list[int]:
        # randomly select one sample from each cluster
        selected_indices = []
        for i in range(self.k):
            cluster_indices = np.where(self.kmeans.labels_ == i)[0]
            if cluster_indices.size == 0:
                continue
            selected_index = np.random.choice(cluster_indices)
            selected_indices.append(selected_index)
        
        # randomly select l samples from selected indices
        l = min(l, len(selected_indices))
        return np.random.choice(selected_indices, size=l, replace=False).tolist()