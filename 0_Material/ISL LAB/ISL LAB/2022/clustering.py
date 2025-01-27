import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure

def get_data_blobs(n_points):
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=n_points, centers=5, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def build_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X.numpy())
    return kmeans

def assign_kmeans(km, X):
    y_pred = torch.tensor(km.predict(X.numpy()), dtype=torch.long)
    return y_pred

def compare_clusterings(ypred_1, ypred_2):
    h_match, c_match, v_match = homogeneity_completeness_v_measure(ypred_1.numpy(), ypred_2.numpy())
    h_nonmatch, c_nonmatch, v_nonmatch = homogeneity_completeness_v_measure(ypred_1.numpy(), ypred_2.numpy())
    return h_match, c_match, v_match, h_nonmatch, c_nonmatch, v_nonmatch
