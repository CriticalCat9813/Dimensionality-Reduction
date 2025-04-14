import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags, lil_matrix
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

# ... (Previous imports and SafeTSNE class remain the same)

print("Loading and preparing data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

np.random.seed(42)
landmark_idx = np.random.choice(len(X), 6000, replace=False)
landmarks = X[landmark_idx].copy()
landmark_labels = y[landmark_idx]

# Create mapping between original indices and subset indices
landmark_id_to_subset_idx = {orig_idx: subset_idx 
                           for subset_idx, orig_idx in enumerate(landmark_idx)}

# Explicit probability calculation with sparse matrices
print("\nCalculating explicit probabilities...")
nbrs = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(X)
P_explicit = lil_matrix((6000, 6000), dtype=np.float32)

for i in range(6000):
    distances, indices = nbrs.kneighbors(landmarks[i:i+1])
    valid_mask = np.isin(indices[0], landmark_idx)
    valid_indices = indices[0][valid_mask]
    
    if len(valid_indices) == 0:
        P_explicit[i, i] = 1.0
        continue
        
    # Convert original indices to subset indices
    subset_indices = [landmark_id_to_subset_idx[idx] for idx in valid_indices 
                     if idx in landmark_id_to_subset_idx]
    
    # Binary search for perplexity calibration with stability fixes 🚨
    beta = 1.0
    beta_min, beta_max = 1e-10, 1e10
    target_entropy = np.log(30)
    last_entropy = None
    
    for _ in range(100):  # Increased iterations for convergence
        with np.errstate(over='ignore', under='ignore'):
            exponents = -beta * distances[0][valid_mask]**2
            exponents = np.clip(exponents, -700, 700)  # Prevent overflow
            prob = np.exp(exponents)
        
        prob_sum = np.maximum(prob.sum(), 1e-300)
        prob /= prob_sum
        
        entropy = -np.sum(prob * np.log(prob + 1e-300))  # Use natural log
        if np.isclose(entropy, target_entropy, rtol=1e-3) or last_entropy == entropy:
            break
        last_entropy = entropy
        
        if entropy > target_entropy:
            beta_min = beta
            beta = beta_max if beta_max < 1e20 else (beta * 10)
        else:
            beta_max = beta
            beta = beta_min if beta_min > 1e-20 else (beta / 10)
    
    # Store probabilities with validation 🚨
    total_prob = 0.0
    for idx, p in zip(subset_indices, prob):
        if 0 <= idx < 6000 and p > 1e-20:
            P_explicit[i, idx] = p
            total_prob += p
    if total_prob < 0.99:  # Add residual to self
        P_explicit[i, i] += 1.0 - total_prob

# Symmetrize and normalize probability matrix 🚨
P_explicit = 0.5 * (P_explicit + P_explicit.T)
P_explicit = P_explicit.tocsr()
P_explicit.data = np.clip(P_explicit.data, 1e-12, None)
row_sums = P_explicit.sum(axis=1).A.ravel()
P_explicit = P_explicit.multiply(1 / np.maximum(row_sums, 1e-12)[:, np.newaxis])

def invert_sparse_probabilities(P):
    """Safely convert probability sparse matrix to distance matrix"""
    P = P.tocsr()
    P.data = 1.0 / np.clip(P.data, 1e-12, None)  # Prevent division by zero
    return P.toarray()

P_explicit_dist = invert_sparse_probabilities(P_explicit)



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
try:
    solver = splu(L_NN)
except RuntimeError as e:
    print(f"Factorization failed. Condition number estimate: {1/np.linalg.cond(L_NN.todense())}")
    raise

# Solve for analytical probabilities
P_analytical = lil_matrix((6000, 6000), dtype=np.float32)

print("Solving linear systems...")
for i in range(6000):
    b = -B[:, i].toarray().ravel().astype(np.float32)
    
    try:
        x = solver.solve(b)
    except RuntimeError:
        x = np.zeros(L_NN.shape[0])
    
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

P_analytical_dist = invert_sparse_probabilities(P_analytical)

# Validation checks
assert np.allclose(np.array(P_explicit.sum(axis=1)).ravel(), 1.0, atol=1e-6), "Explicit P not valid"
assert np.allclose(np.array(P_analytical.sum(axis=1)).ravel(), 1.0, atol=1e-6), "Analytical P not valid"



# Validation checks for distance matrices 
assert np.all(np.isfinite(P_explicit_dist)), "Explicit distances contain invalid values"
assert np.all(np.isfinite(P_analytical_dist)), "Analytical distances contain invalid values"



def plot_embeddings(emb1, emb2, labels, filename):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    # Common visualization parameters
    plot_params = {
        'c': labels,
        'cmap': 'tab10',
        's': 8,
        'alpha': 0.7,
        'edgecolor': 'none'
    }
    
    axs[0].scatter(emb1[:, 0], emb1[:, 1], **plot_params)
    axs[0].set_title("Explicit Probability Method\n(Standard t-SNE)", fontsize=12)
    
    axs[1].scatter(emb2[:, 0], emb2[:, 1], **plot_params)
    axs[1].set_title("Analytical Probability Method\n(Graph Laplacian)", fontsize=12)
    
    # Add colorbar between subplots
    cbar = fig.colorbar(axs[0].collections[0], ax=axs, shrink=0.95)
    cbar.set_label('Digit Class', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Modified TSNE configuration 🚨
tsne_config = {
    'n_components': 2,
    'perplexity': 30,
    'early_exaggeration': 12,
    'learning_rate': 'auto',
    'n_iter': 1000,
    'metric': 'precomputed',
    'init': 'random',  # Changed from 'pca'
    'random_state': 42,
    'n_jobs': -1
}


print("\nRunning stabilized t-SNE...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Computing explicit embedding...")
    embedding_explicit = SafeTSNE(**tsne_config).fit_transform(P_explicit_dist)
    
    print("Computing analytical embedding...")
    embedding_analytical = SafeTSNE(**tsne_config).fit_transform(P_analytical_dist)

# Generate final plot
plot_embeddings(embedding_explicit, embedding_analytical, 
                landmark_labels, 'random_mnist.png')