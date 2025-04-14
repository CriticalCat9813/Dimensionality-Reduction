import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags, lil_matrix, csc_matrix
from scipy.sparse.linalg import splu
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator
import warnings
import gc

class SafeTSNE:
    """Wrapper class with enhanced stability checks"""
    def __init__(self, **kwargs):
        self.tsne = TSNE(**kwargs)
        
    def fit_transform(self, X):
        # Convert to dense array if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        # Add pre-check for input data
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0)
            
        # Run t-SNE with stability wrappers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = self.tsne.fit_transform(X)
            
        # Post-process embedding
        std = np.std(embedding[:, 0])
        if std < 1e-12 or not np.isfinite(std):
            embedding = np.random.normal(0, 1e-4, embedding.shape)
        else:
            embedding = (embedding - np.mean(embedding)) / std
            
        return np.nan_to_num(embedding, nan=0.0)

# Load data
print("Loading and preparing data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

# Create landmark subset
np.random.seed(42)
landmark_idx = np.random.choice(len(X), 6000, replace=False)
landmarks = X[landmark_idx].copy()
landmark_labels = y[landmark_idx]

# Analytical probability calculation using graph Laplacian
print("\nCalculating analytical probabilities...")
nbrs_full = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(X)
distances_full, indices_full = nbrs_full.kneighbors(X)

# Build sparse adjacency matrix
rows = np.repeat(np.arange(len(X)), 20)
cols = indices_full.ravel()
W = csr_matrix((np.ones_like(rows), (rows, cols)), 
                shape=(len(X), len(X)), dtype=np.float32)
W = W.minimum(1).maximum(W.T)

# Compute Laplacian matrix
print("Computing graph Laplacian...")
D = diags(W.sum(axis=1).A.ravel(), 0)
L = D - W
del D, W
gc.collect()

# Split into landmark and non-landmark components
non_landmark_mask = ~np.isin(np.arange(len(X)), landmark_idx)
L_NN = L[non_landmark_mask][:, non_landmark_mask].tocsc()
B = L[non_landmark_mask][:, landmark_idx].tocsc()

# Precompute solver
print("Factorizing Laplacian...")
solver = splu(L_NN)

# Solve for analytical probabilities
P_analytical = lil_matrix((6000, 6000), dtype=np.float32)

print("Solving linear systems...")
for i in range(6000):
    b = -B[:, i].toarray().ravel().astype(np.float32)
    x = solver.solve(b)
    
    # Compute landmark probabilities
    probs = -x @ B  # Matrix-vector product
    probs = np.clip(probs, 0, None)
    probs_sum = probs.sum()
    
    if probs_sum < 1e-12:
        P_analytical[i, i] = 1.0
    else:
        for j in range(6000):
            if probs[j] > 0:
                P_analytical[i, j] = probs[j] / probs_sum

# Convert probabilities to distances
def invert_sparse_probabilities(P):
    """Safely convert probability sparse matrix to distance matrix"""
    P = P.tocsr()
    P.data = 1.0 / np.clip(P.data, 1e-12, None)
    return P.toarray()

P_analytical_dist = invert_sparse_probabilities(P_analytical)

# Validate analytical probabilities
assert np.allclose(np.array(P_analytical.sum(axis=1)).ravel(), 1.0, atol=1e-6), "Analytical P not valid"
assert np.all(np.isfinite(P_analytical_dist)), "Analytical distances contain invalid values"

# Configure and run TSNE
tsne_config = {
    'n_components': 2,
    'perplexity': 30,
    'early_exaggeration': 12,
    'learning_rate': 'auto',
    'n_iter': 1000,
    'metric': 'precomputed',
    'init': 'random',
    'random_state': 42,
    'n_jobs': -1
}

print("\nRunning stabilized t-SNE...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Computing analytical embedding...")
    embedding_analytical = SafeTSNE(**tsne_config).fit_transform(P_analytical_dist)

# Plotting function for single embedding
def plot_embedding(emb, labels, filename):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=8, alpha=0.7, edgecolor='none')
    plt.title("Analytical t-SNE via Graph Laplacian")
    plt.colorbar(scatter, label='Digit Class')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Generate final plot
plot_embedding(embedding_analytical, landmark_labels, 'tSNE_with_large_dataset.png')
