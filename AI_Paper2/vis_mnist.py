import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import os
import cvxpy as cp

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

np.random.seed(42)
indices = np.random.choice(len(X), 6000, replace=False)
X_subset = X[indices]
y_subset = y[indices]


pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_subset)


perplexities = [5, 10, 30, 50, 100]
tsne_results = {}
for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_jobs=-1)
    tsne_results[perp] = tsne.fit_transform(X_pca)

isomap = Isomap(n_components=2, n_neighbors=12, n_jobs=-1)
X_isomap = isomap.fit_transform(X_pca)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=12, method='standard', reg=0.01, random_state=42)
X_lle = lle.fit_transform(X_pca)

spectral = SpectralEmbedding(n_components=2, n_neighbors=12, affinity='nearest_neighbors', random_state=42)
X_spectral = spectral.fit_transform(X_pca)


def sammon(X, n_out=2, max_iter=500, alpha=0.1, eps=1e-12, verbose=True):
    """Numerically stable Sammon mapping"""
    n = X.shape[0]
    D = squareform(pdist(X)) + eps
    stress_scale = 1 / D.sum()
    
    Y = PCA(n_components=n_out).fit_transform(X)
    Y = (Y - Y.mean(0)) / Y.std(0)
    
    for it in range(max_iter):
        D_hat = squareform(pdist(Y)) + eps
        
        # Gradient descent
        valid_mask = (D_hat * D) > eps
        ratio = np.zeros_like(D)
        np.divide(D - D_hat, D_hat * D, where=valid_mask, out=ratio)
        
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 2 * stress_scale * np.nansum(ratio[:, i, None] * diff, axis=0)
        
        # Adaptive learning rate
        alpha_t = alpha * (1 - it/max_iter)
        Y -= alpha_t * grad
        
        # if verbose and (it % 100 == 0):
        #     stress = stress_scale * np.nansum((D - D_hat)**2 / D)
        #     print(f"Iter {it}: Stress {stress:.4f}")
    
    return Y

X_sammon = sammon(X_pca)
# def mvu(X, n_components=2, n_neighbors=5):
#     """Maximum Variance Unfolding with stability checks"""
#     n = X.shape[0]
    
#     # Input validation
#     if np.any(np.isnan(X)):
#         raise ValueError("Input contains NaN values")
#     if len(np.unique(X, axis=0)) < n:
#         raise ValueError("Duplicate samples in input data")
    
#     # Create neighborhood graph
#     knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
#     K = knn.kneighbors_graph(X).toarray()
    
#     # Create optimization problem
#     G = cp.Variable((n, n), symmetric=True)
#     objective = cp.Maximize(cp.trace(G))
    
#     # Constraints with numerical stability
#     constraints = [
#         G >> 1e-6 * np.eye(n),  # Ensure positive definiteness
#         cp.sum(G, axis=0) == 0,  # Centering constraint
#         *[cp.abs(G[i,i] + G[j,j] - 2*G[i,j] - np.sum((X[i]-X[j])**2)) <= 1e-3
#           for i in range(n)
#           for j in np.where(K[i])[0]]
#     ]
    
#     # Solve with enhanced parameters
#     prob = cp.Problem(objective, constraints)
#     try:
#         prob.solve(solver=cp.SCS, 
#                  eps=1e-5, 
#                  max_iters=10000, 
#                  verbose=False)
#     except Exception as e:
#         raise RuntimeError(f"MVU optimization failed: {str(e)}")
    
#     # Check and handle solution status
#     if G.value is None or np.any(np.isnan(G.value)):
#         return np.zeros((n, n_components))  # Fallback to zeros
        
#     # Stable PCA transformation
#     G_clean = np.nan_to_num(G.value, nan=0.0)
#     G_clean = 0.5 * (G_clean + G_clean.T)  # Ensure symmetry
#     return PCA(n_components=n_components, 
#              svd_solver='arpack').fit_transform(G_clean)

# X_pca_centered = (X_pca[:100] - X_pca[:100].mean(0)) / X_pca[:100].std(0)
# X_mvu = mvu(X_pca_centered)
# y_mvu = y_subset[:100]
output_dir = "mnist_dim_reduction_plots"
os.makedirs(output_dir, exist_ok=True)

# Modified visualization section
methods = [
    ("t-SNE (perplexity=5)", tsne_results[5], y_subset),
      ("t-SNE (perplexity=10)", tsne_results[10], y_subset),
    ("t-SNE (perplexity=30)", tsne_results[30], y_subset),
    ("t-SNE (perplexity=50)", tsne_results[50], y_subset),
    ("t-SNE (perplexity=100)", tsne_results[100], y_subset),
    ("Isomap (n_neighbors=12)", X_isomap, y_subset),
    ("LLE (n_neighbors=12)", X_lle, y_subset),
    ("Laplacian Eigenmaps (n_neighbors=12)", X_spectral, y_subset),
    ("Sammon Mapping (max_iter=100)", X_sammon, y_subset)
]

for method_name, embedding, labels in methods:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.title(method_name, fontsize=14)
    plt.axis('off')
    
    filename = method_name.replace("(", "").replace(")", "").replace(" ", "_").replace("=", "")
    filename = f"{filename}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")

print("\nAll plots saved in:", os.path.abspath(output_dir))