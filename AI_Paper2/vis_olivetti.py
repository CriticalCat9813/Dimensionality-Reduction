import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from scipy.spatial.distance import pdist, squareform
import os


olivetti = fetch_olivetti_faces()
X = olivetti.data.astype(np.float32)
y = olivetti.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)

methods = []

# t-SNE with different perplexities
perplexities = [5, 15, 30, 50, 100]
for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_jobs=-1)
    methods.append((f"t-SNE (perp={perp})", tsne.fit_transform(X_pca)))


methods += [
    ("Isomap (k=12)", Isomap(n_components=2, n_neighbors=12).fit_transform(X_pca)),
    ("LLE (k=12)", LocallyLinearEmbedding(n_components=2, n_neighbors=12, reg=0.1).fit_transform(X_pca)),
    ("Laplacian (k=12)", SpectralEmbedding(n_components=2, n_neighbors=12).fit_transform(X_pca))
]

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
methods.append(("Sammon Mapping", sammon(X_pca)))

output_dir = 'olivetti_embeddings'
os.makedirs(output_dir, exist_ok=True)


n_methods = len(methods)
n_cols = 3
n_rows = (n_methods + n_cols - 1) // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
axs = axs.flatten()

# Get colormap using updated method
cmap = colormaps.get_cmap('gist_ncar').resampled(40)

for idx, (name, emb) in enumerate(methods):
    scatter = axs[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap,
                             s=15, alpha=0.8, edgecolor='w', linewidth=0.3)
    axs[idx].set_title(name, fontsize=12)
    axs[idx].axis('off')


for idx in range(n_methods, len(axs)):
    fig.delaxes(axs[idx])

# Add colorbar
cbar = fig.colorbar(scatter, ax=axs[:n_methods], 
                    orientation='horizontal', 
                    fraction=0.02, pad=0.02)
cbar.set_label('Subject ID', fontsize=10)
cbar.set_ticks(np.linspace(0, 39, 40))
cbar.set_ticklabels([str(i+1) for i in range(40)])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_view.png'), dpi=200, bbox_inches='tight')
plt.close()

# Create individual plots
os.makedirs(os.path.join(output_dir, 'individual'), exist_ok=True)

for name, emb in methods:
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap,
                s=20, alpha=0.8, edgecolor='w', linewidth=0.3)
    plt.title(name, fontsize=14)
    plt.axis('off')
    
    filename = f"olivetti_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(output_dir, 'individual', filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"Visualizations saved to: {os.path.abspath(output_dir)}")