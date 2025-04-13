import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.fftpack import dct, idct

def add_salt_and_pepper(X, noise_prob=0.2, random_state=42):
    """
    Adds salt-and-pepper noise to each pixel in X with probability noise_prob.
    X is assumed to be in [0,1].
      - "Salt" flips a pixel to 1.0
      - "Pepper" flips a pixel to 0.0
    Probability of flipping to 1 vs. 0 is each 0.5 * noise_prob.
    """
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()
    flip_mask = rng.random(X_noisy.shape) < noise_prob  # which pixels to flip
    salt_mask = rng.random(X_noisy.shape) < 0.5
    X_noisy[flip_mask & salt_mask] = 1.0
    X_noisy[flip_mask & ~salt_mask] = 0.0
    return X_noisy


def apply_pca(X, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_reduced)
    return X_recon

def apply_rp_gaussian(X, n_components):
   
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_reduced = rp.fit_transform(X)
    G = rp.components_
    if hasattr(G, "toarray"):
        G = G.toarray()
    G = G.T  # shape (n_features, n_components)
    G_pinv = np.linalg.pinv(G)
    X_recon = X_reduced @ G_pinv
    return X_recon

def apply_srp_achlioptas(X, n_components):
    n_samples, d = X.shape
    probs = np.random.choice([+1, 0, -1], size=(d, n_components), p=[1/6, 2/3, 1/6])
    R = np.sqrt(3) * probs
    X_reduced = X @ R
    R_pinv = np.linalg.pinv(R)
    X_recon = X_reduced @ R_pinv
    return X_recon

def apply_dct(X, n_components, original_dim=784):
    
    side = int(np.sqrt(original_dim))  # 28 for EMNIST
    X_recon = np.zeros_like(X)
    for i in range(X.shape[0]):
        img = X[i].reshape(side, side)
        dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
        # Keep n_components by Manhattan distance from (0,0)
        coords = np.indices((side, side)).reshape(2, -1).T
        distances = np.sum(coords, axis=1)
        idx = np.argsort(distances)[:n_components]
        mask = np.zeros(side*side, dtype=bool)
        mask[idx] = True
        flat = dct_img.flatten()
        dct_thresh = np.zeros_like(flat)
        dct_thresh[mask] = flat[mask]
        dct_thresh = dct_thresh.reshape(side, side)
        img_recon = idct(idct(dct_thresh.T, norm='ortho').T, norm='ortho')
        X_recon[i] = img_recon.flatten()
    return X_recon


def pairwise_distance_error(X_orig, Xhat, n_pairs=100, R=None, random_state=42):
    rng = np.random.default_rng(random_state)
    errors = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(X_orig), size=2, replace=False)
        dist_orig = np.linalg.norm(X_orig[i] - X_orig[j])
        if R is not None:
            # For RP methods: compute projected distance and scale it:
            k = R.shape[1]
            y_i = X_orig[i] @ R
            y_j = X_orig[j] @ R
            dist_proj = np.linalg.norm(y_i - y_j)
            dist_scaled = np.sqrt(d/k) * dist_proj
            errors.append(abs(dist_orig - dist_scaled))
        else:
            # For reconstruction methods:
            dist_recon = np.linalg.norm(Xhat[i] - Xhat[j])
            errors.append(abs(dist_orig - dist_recon))
    return np.mean(errors)

if __name__ == "__main__":
    
    emnist = fetch_openml(data_id=40996, as_frame=False)
    X_all = emnist.data.astype(np.float64) / 255.0

    # For a quick test set
    _, X_test = train_test_split(X_all, test_size=0.2, random_state=42)
    n_samples, d = X_test.shape
    print("Test set shape:", X_test.shape)

    # Add salt-and-pepper noise with probability 0.2
    X_noisy = add_salt_and_pepper(X_test, noise_prob=0.2, random_state=999)
    print("Noisy dataset created. Example pixel range:", X_noisy.min(), X_noisy.max())

   
    base_dist_err = pairwise_distance_error(X_test, X_noisy, n_pairs=100)
    print(f"Distance error from original to noisy images: {base_dist_err:.4f}")

    methods = {
        'RP (Gaussian)': apply_rp_gaussian,
        'SRP (Achlioptas)': apply_srp_achlioptas,
        'PCA': apply_pca,
        'DCT': lambda X, k: apply_dct(X, k, original_dim=d)
      }

    dims = [50, 100, 200, 300, 400, 500, 600, 700]
    results = {m: [] for m in methods}

    for k in dims:
        print(f"\nReduced dimension = {k}")
        for method_name, func in methods.items():
            X_recon = func(X_noisy, k)
            dist_err = pairwise_distance_error(X_noisy, X_recon, n_pairs=100)
            results[method_name].append(dist_err)
            print(f"  {method_name} -> Dist Error = {dist_err:.4f}")

    plt.figure(figsize=(8,5))
    for method_name in methods:
        plt.plot(dims, results[method_name], marker='o', label=method_name)
    plt.xlabel("Reduced Dimension (k)")
    plt.ylabel("Pairwise Distance Error (Noisy Data)")
    plt.title("Distance Error between Noisy Images\n(averaged over 100 random pairs)")
    plt.legend()
    plt.grid(True)
    plt.show()
