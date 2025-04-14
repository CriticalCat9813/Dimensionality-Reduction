import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


newsgroups = fetch_20newsgroups(subset='train', categories= None, shuffle=True, random_state=42)

print(f"Number of documents: {len(newsgroups.data)}")

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
X_full = vectorizer.fit_transform(newsgroups.data)  # shape (n_docs, n_features)

n_samples, n_features = X_full.shape
print(f"TF-IDF matrix shape: {X_full.shape}")


n_subset = 2000
if n_samples > n_subset:
    X_full = X_full[:n_subset]
    n_samples = n_subset

# Convert to a dense array if needed for dot products
# (This can be memory-heavy for large sets, but we do it for demonstration.)
X_dense = X_full.toarray()

def inner_product_error(X_orig, X_recon, n_pairs=100, random_state=42):
    """
    Randomly choose n_pairs of document vectors (i, j),
    compute the difference in their inner product:
      error_ij = |(x_i dot x_j) - (x'_i dot x'_j)|
    Return the average of these errors.
    """
    rng = np.random.default_rng(random_state)
    errors = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(X_orig), size=2, replace=False)
        orig_ip = np.dot(X_orig[i], X_orig[j])
        recon_ip = np.dot(X_recon[i], X_recon[j])
        errors.append(abs(orig_ip - recon_ip))
    return np.mean(errors)

def apply_rp(X, n_components):
    """
    Gaussian Random Projection with reconstruction by pseudo-inverse
    to get the new representation (like a direct transform).
    """
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_proj = rp.fit_transform(X)
   
    # We'll interpret X_proj as the final representation for the inner product check
    return X_proj

def apply_svd(X, n_components):
   
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)
    return X_svd

# ---------------------------------------------------------
# 4. EXPERIMENT
# ---------------------------------------------------------
# We'll vary k in some range, e.g. [50, 100, 200, 300, 400, 500].
# For each k, we measure the average error in the inner products
# for both RP and SVD.

dims = [50, 100, 200, 300, 400, 500]
n_pairs = 100

rp_errors = []
svd_errors = []

for k in dims:
    print(f"\nReduced dimension = {k}")

    # Random Projection
    X_rp = apply_rp(X_dense, k)
    rp_err = inner_product_error(X_dense, X_rp, n_pairs=n_pairs, random_state=42)
    rp_errors.append(rp_err)
    print(f"  RP (Gaussian) error = {rp_err:.4f}")

    X_svd = apply_svd(X_full, k)  # can pass sparse to TruncatedSVD directly
    # For a fair comparison, convert to dense if needed:
    X_svd_dense = X_svd if isinstance(X_svd, np.ndarray) else X_svd.toarray()
    svd_err = inner_product_error(X_dense, X_svd_dense, n_pairs=n_pairs, random_state=42)
    svd_errors.append(svd_err)
    print(f"  SVD error = {svd_err:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(dims, rp_errors, marker='o', label="RP (Gaussian)")
plt.plot(dims, svd_errors, marker='s', label="SVD")
plt.xlabel("Reduced Dimension (k)")
plt.ylabel("Avg. Inner Product Error")
plt.title("Inner Product Error on 20 Newsgroups")
plt.grid(True)
plt.legend()
plt.show()
