import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from scipy.spatial.distance import pdist, squareform
import os
import cv2

base_folder = 'Coil-20'

images = []
labels = []

# Sort folders numerically
folder_names = sorted(os.listdir(base_folder), key=lambda x: int(x) if x.isdigit() else 0)

for label_str in folder_names:
    subfolder_path = os.path.join(base_folder, label_str)
    
    if not os.path.isdir(subfolder_path):
        continue
    
    try:
        label = int(label_str) - 1  # Convert to 0-based index
    except ValueError:
        continue

    # Load images with error handling
    for filename in sorted(os.listdir(subfolder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())
                labels.append(label)

# Convert to numpy arrays
X = np.array(images, dtype=np.float32) / 255.0
y = np.array(labels)

print(f"Dataset shape: {X.shape}")
print(f"Unique labels: {np.unique(y)}")


pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)

# Configure methods with optimized parameters
methods = []

# t-SNE variations
perplexities = [5, 10, 30, 50, 100]
for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_jobs=-1)
    methods.append((f"t-SNE (perp={perp})", tsne.fit_transform(X_pca)))

# Isomap with increased neighbors
methods.append(("Isomap (k= 50)", Isomap(n_components=2, n_neighbors= 50, n_jobs=-1).fit_transform(X_pca)))

# LLE with regularization
methods.append(("LLE (k= 50)", LocallyLinearEmbedding(n_components=2, n_neighbors=50,
                                                   reg=0.1, random_state=42).fit_transform(X_pca)))

# Laplacian Eigenmaps
methods.append(("Laplacian (k= 50)", SpectralEmbedding(n_components=2, n_neighbors=50,
                                                    n_jobs=-1,
                                                    affinity='nearest_neighbors', 
                                                    random_state=42).fit_transform(X_pca)))

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

X_sammon = sammon(X_pca, max_iter=500, alpha=0.5)
methods.append(("Sammon Mapping", X_sammon))

output_dir = 'coil20_embeddings'
os.makedirs(output_dir, exist_ok=True)

for name, emb in methods:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], 
                         c=y, cmap='tab20', 
                         s=25, alpha=0.8,
                         edgecolor='w', linewidth=0.3)
    
    # Formatting
    plt.title(name, fontsize=14, pad=12)
    plt.axis('off')
    
    # Add colorbar to each plot
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Object Class', fontsize=10)
    cbar.set_ticks(np.linspace(0, 19, 20))
    cbar.set_ticklabels([str(i+1) for i in range(20)])  # Original COIL labels 1-20
    
    # Save individual files
    filename = (name.replace(' ', '_')
                .replace('(', '').replace(')', '')
                .replace('=', '_') + '.png')
    
    plt.savefig(os.path.join(output_dir, filename), 
               dpi=300, 
               bbox_inches='tight',
               pad_inches=0.1)
    plt.close() 

print(f"Visualizations saved to: {os.path.abspath(output_dir)}")