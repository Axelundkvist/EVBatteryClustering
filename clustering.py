import numpy as np
import random
from typing import Dict, List, Tuple

def k_means_clustering(data: List[np.ndarray], k: int, max_iter: int = 100) -> List[int]:
    """
    Performs K-means clustering on a list of feature vectors.
    :param data: List of numpy arrays, each array is a feature vector.
    :param k: Number of clusters.
    :param max_iter: Maximum number of iterations.
    :return: List of cluster labels corresponding to each feature vector.
    """
    n_samples = len(data)
    # Initialize centroids by randomly choosing k unique data points.
    initial_indices = random.sample(range(n_samples), k)
    centroids = [data[i].copy() for i in initial_indices]
    
    labels = [None] * n_samples
    for _ in range(max_iter):
        new_labels = []
        # Assignment step
        for vec in data:
            distances = [np.linalg.norm(vec - centroid) for centroid in centroids]
            new_labels.append(int(np.argmin(distances)))
        
        # Check for convergence (if labels do not change)
        if new_labels == labels:
            break
        labels = new_labels
        
        # Update step: recompute centroids as mean of points in each cluster.
        for i in range(k):
            # Get all vectors assigned to cluster i.
            cluster_points = [data[j] for j in range(n_samples) if labels[j] == i]
            if cluster_points:  # avoid division by zero
                centroids[i] = np.mean(cluster_points, axis=0)
    
    return labels

# Example usage within your server_functions framework:
def cluster_clients(client_updates: Dict[str, Tuple[List[np.ndarray], Dict]]) -> Dict[str, int]:
    """
    Given client updates where metadata already includes the six features,
    cluster the clients using K-means and return a mapping from client ID to cluster label.
    :param client_updates: Dict mapping client ID to a tuple where the second element
                           is a metadata dictionary containing the features.
    :return: Dict mapping client ID to cluster label.
    """
    client_ids = list(client_updates.keys())
    # Extract feature vectors from metadata; assume keys: "V_mean", "V_std", "V_kur", "V_sk", "T_mean", "S_mean"
    feature_vectors = []
    for cid in client_ids:
        metadata = client_updates[cid][1]
        # Create a feature vector in a fixed order.
        vec = np.array([
            metadata["V_mean"],
            metadata["V_std"],
            metadata["V_kur"],
            metadata["V_sk"],
            metadata["T_mean"],
            metadata["S_mean"]
        ], dtype=float)
        feature_vectors.append(vec)
    
    # Choose the number of clusters; for example, k=3 (or you could decide based on your domain knowledge)
    k = 3
    labels = k_means_clustering(feature_vectors, k)
    
    # Map each client to its cluster label.
    cluster_map = {cid: label for cid, label in zip(client_ids, labels)}
    return cluster_map

# Example call (assuming client_updates is provided from train.py):
# client_clusters = cluster_clients(client_updates)
