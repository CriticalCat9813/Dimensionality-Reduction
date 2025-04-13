import numpy as np
import matplotlib.pyplot as plt
import math, time
from statistics import mean, stdev
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.fftpack import dct, idct


print("Loading EMNIST Digits from OpenML (data_id=40996)...")
emnist = fetch_openml(data_id=40996, as_frame=False)
X_all = emnist.data.astype(np.float64) / 255.0  # Normalize to [0,1]
_, X_test = train_test_split(X_all, test_size=0.2, random_state=42)
n_samples, d = X_test.shape
print("Test set shape:", X_test.shape)

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_reduced)
    return X_recon

def apply_dct(X, n_components):
    side = int(np.sqrt(d))  # should be 28 for EMNIST
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


def rp_gaussian_projection(X, n_components, random_state=None):
    rng = np.random.default_rng(random_state)
    R = rng.standard_normal((d, n_components))  # d x k matrix with Gaussian entries
    Y = X @ R
    return Y, R

def rp_achlioptas_projection(X, n_components, random_state=None):
    rng = np.random.default_rng(random_state)
    R = np.sqrt(3) * rng.choice([+1, 0, -1], size=(d, n_components), p=[1/6, 2/3, 1/6])
    Y = X @ R
    return Y, R

def reconstruct_from_projection(Y, R):
    R_pinv = np.linalg.pinv(R)
    X_recon = Y @ R_pinv
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


def confidence_interval(values, alpha=0.95):
    z = 1.96
    m = mean(values)
    s = stdev(values)
    n = len(values)
    ci = z * (s / math.sqrt(n))
    return m, m - ci, m + ci

samples_for_ci = 100  # number of repeated measurements for CI


# For RP methods, we use the scaling: sqrt(d/k) * ||R x_i - R x_j||.
# For PCA and DCT, we compute the error between the original and the reconstruction distances.
methods = {
    'RP (Gaussian)': ("rp", rp_gaussian_projection),
    'SRP (Achlioptas)': ("rp", rp_achlioptas_projection),
    'PCA': ("recon", apply_pca),
    'DCT': ("recon", apply_dct)
}

dims = [50, 100, 200, 300, 400, 500, 600, 700]

# We'll collect the average pairwise distance error and compute 95% CI for each method and each dimension.
results = {m: [] for m in methods}
cis = {m: [] for m in methods}
runtime = {m : [] for m in methods}
n_runs = 100


for k in dims:
    print(f"\nReduced dimension = {k}")
    for method_name, (mode, func) in methods.items():
        sample_errors = []
        times = []
        for _ in range(n_runs):
         start = time.perf_counter()
         if mode == "rp":
            Y, R = func(X_test, k, random_state=None)
            # We include reconstruction as part of the cost
            _ = reconstruct_from_projection(Y, R)
         else:
            _ = func(X_test, k)
         end = time.perf_counter()
         times.append(end - start)
        avg_time = np.mean(times)
        runtime[method_name].append(avg_time)
        print(f"  {method_name}: avg runtime = {avg_time:.4f} sec")
        for _ in range(samples_for_ci):
            if mode == "rp":
                Y, R = func(X_test, k, random_state=None)
                # For RP methods, we don't reconstruct; we compute the scaled distance error directly.
                err = pairwise_distance_error(X_test, None, n_pairs=100, R=R, random_state=None)
            else:
                Xhat = func(X_test, k)
                err = pairwise_distance_error(X_test, Xhat, n_pairs=100, R=None, random_state=None)
            sample_errors.append(err)
        m_err, ci_low, ci_high = confidence_interval(sample_errors)
        results[method_name].append(m_err)
        cis[method_name].append((ci_low, ci_high))
        print(f"  {method_name}: Avg Dist Error = {m_err:.6f}, CI = [{ci_low:.6f}, {ci_high:.6f}]")

# ---------------------------------------------------

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for method_name in methods:
    means = np.array(results[method_name])
    ci_vals = np.array(cis[method_name])
    yerr = np.vstack([means - ci_vals[:, 0], ci_vals[:, 1] - means])
    plt.errorbar(dims, means, yerr=yerr, marker='o', label=method_name)
plt.xlabel("Reduced Dimension (k)")
plt.ylabel("Avg Pairwise Distance Error")
plt.title("Pairwise Distance Error (Noisy/No Noise) with 95% CI")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 5))
for method_name in methods:
    plt.plot(dims, runtime[method_name], marker='o', label=method_name)
plt.xlabel("Reduced Dimension (k)")
plt.ylabel("Average Runtime (sec)")
plt.title("Runtime vs. Reduced Dimension\n(Proxy for FLOP Count)")
plt.yscale("log")  # Log scale for runtime
plt.legend()
plt.grid(True)
plt.show()