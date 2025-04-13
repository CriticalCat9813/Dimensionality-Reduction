# Dimensionality-Reduction

# Dimensionality Reduction: Research Paper Implementations

This repository contains Python implementations of three influential dimensionality reduction techniques, each based on a foundational research paper. These implementations aim to reproduce key results, visualize outcomes, and provide an educational reference for those studying or applying dimensionality reduction.

## 📚 Implemented Papers

### 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)
**Paper**: *Visualizing Data using t-SNE*  
**Authors**: Laurens van der Maaten, Geoffrey Hinton  
**Link**: [JMLR, 2008](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)  
**Summary**: A nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data in 2 or 3 dimensions for visualization. Emphasizes local structure while maintaining global clusters through a heavy-tailed Student-t distribution in the embedding space.

---

### 2. Random Projection for Image and Text Data
**Paper**: *Random Projection in Dimensionality Reduction: Applications to Image and Text Data*  
**Authors**: Ella Bingham, Heikki Mannila  
**Link**: [KDD 2001](https://dl.acm.org/doi/10.1145/502512.502546)  
**Summary**: Demonstrates the effectiveness of random projections in reducing dimensionality while preserving pairwise distances, relying on the Johnson–Lindenstrauss lemma. Lightweight and computationally efficient.

---

### 3. C-GMVAE: Gaussian Mixture Variational Autoencoder with Contrastive Learning
**Paper**: *Gaussian Mixture Variational Autoencoder with Contrastive Learning for Multi-Label Classification*  
**Authors**: Junwen Bai, Shufeng Kong, Carla Gomes  
**Link**: [ICML 2022](https://arxiv.org/abs/2112.00976)  
**Summary**: A probabilistic model for multi-label classification using a VAE with a multimodal latent space and contrastive loss to learn label and feature embeddings. Eliminates the need for complex modules like GNNs while achieving high performance with limited data.

---

## 🛠️ Technologies Used
- Python 3.10+
- NumPy, SciPy, Scikit-learn
- PyTorch / TensorFlow (for VAE-based models)
- Matplotlib / Seaborn for visualizations

## 📂 Directory Structure
