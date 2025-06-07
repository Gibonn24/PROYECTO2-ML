import traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA, TruncatedSVD
from collections import deque
import seaborn as sns
import warnings
import os
import joblib
import tempfile
import requests
from io import BytesIO
import psutil
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

import sys

# --- K-means++ init ---
def kmeans_plus_plus_init(X, k, random_state=None):
    np.random.seed(random_state)
    n_samples, _ = X.shape
    centers = []
    centers.append(X[np.random.randint(n_samples)])
    for _ in range(1, k):
        dist_sq = np.min(np.linalg.norm(X[:, None] - np.array(centers)[None, :], axis=2)**2, axis=1)
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()
        idx = np.searchsorted(cumulative_probs, r)
        centers.append(X[idx])
    return np.array(centers)

# --- K-means completo ---
def kmeans(X, k, max_iter=100, tol=1e-4, random_state=None):
    centers = kmeans_plus_plus_init(X, k, random_state)
    labels = np.zeros(X.shape[0], dtype=int)

    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[new_labels == i].mean(axis=0) if np.any(new_labels == i) else centers[i] for i in range(k)])

        if np.all(labels == new_labels) or np.linalg.norm(new_centers - centers) < tol:
            break

        labels = new_labels
        centers = new_centers

    return labels, centers




def load_data():
    # Leer y juntar todos los .pkl divididos
    chunk_paths = sorted(glob.glob("data/features_part_*.pkl"))
    dataframes = [pd.read_pickle(path) for path in chunk_paths]
    df_total = pd.concat(dataframes, ignore_index=True)
    return df_total

df_full = load_data()
# --- Reduce dimensionalidad (50) ---
X = df_full.drop(columns=["imdbId"])
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# --- K-means (k = 10) ---

X_clust = X_pca

k = 10
labels, centers = kmeans(X_clust, k, random_state=42)


X = X_pca  # o usa X_svd si prefieres
out = {
    "X_pca": X_pca.astype(np.float32),   # liviano
    "imdbId": df_full["imdbId"].values,
    "labels": labels.astype(np.int8),
    "centers": centers.astype(np.float32),
    "pca_mean": pca.mean_.astype(np.float32),
    "pca_components": pca.components_.astype(np.float32),
}

joblib.dump(out, "dataset_features_pca.pkl", compress=3)
print("✅ Guardado dataset_features_pca.pkl (≈15 MB)")



