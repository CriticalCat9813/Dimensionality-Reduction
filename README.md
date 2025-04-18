# Dimensionality-Reduction

## Authors
- [Aarav Desai](https://github.com/CriticalCat9813)
- [Abhinav Goyal](https://github.com/Abhinavg00)
- [Seetha Abhinav](https://github.com/Seetha46)
- [Piyush Kumar](https://github.com/Spiyush1806)
  
## Dimensionality Reduction: Implementations of t-SNE, Random Projection, and C-GMVAE

This repository presents implementations and analyses of three prominent dimensionality reduction techniques, each grounded in foundational research papers. The goal is to provide clear, practical examples that facilitate understanding and application of these methods in various data science and machine learning contexts.​

## Implemented Techniques


### 1. Random Projection
**Paper**: *Random Projection in Dimensionality Reduction: Applications to Image and Text Data*  
**Authors**: Ella Bingham, Heikki Mannila  
**Link**: [KDD 2001](https://dl.acm.org/doi/10.1145/502512.502546)  
**Summary**: Demonstrates the effectiveness of random projections in reducing dimensionality while preserving pairwise distances, relying on the Johnson–Lindenstrauss lemma. Lightweight and computationally efficient.

---
### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
**Paper**: *Visualizing Data using t-SNE*  
**Authors**: Laurens van der Maaten, Geoffrey Hinton  
**Link**: [JMLR, 2008](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)  
**Summary**: A nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data in 2 or 3 dimensions for visualization. Emphasizes local structure while maintaining global clusters through a heavy-tailed Student-t distribution in the embedding space.

---

### 3. C-GMVAE: Gaussian Mixture Variational Autoencoder with Contrastive Learning
**Paper**: *Gaussian Mixture Variational Autoencoder with Contrastive Learning for Multi-Label Classification*  
**Authors**: Junwen Bai, Shufeng Kong, Carla Gomes  
**Link**: [ICML 2022](https://arxiv.org/abs/2112.00976)  
**Summary**: A probabilistic model for multi-label classification using a VAE with a multimodal latent space and contrastive loss to learn label and feature embeddings. Eliminates the need for complex modules like GNNs while achieving high performance with limited data.

**Code based on**: [JunwenBai/C-GMVAE](https://github.com/JunwenBai/C-GMVAE)  
**Original Authors**: Junwen Bai, Shufeng Kong, Carla Gomes
The code in `AI_Paper3/` is adapted for multi-label experiments on MIRFLICKR.

---

## Technologies Used
- Python 3.10+
- NumPy, SciPy, Scikit-learn
- PyTorch / TensorFlow (for VAE-based models)
- Matplotlib / Seaborn for visualizations

## Datasets used
- For Paper 1: 20 newsgroups, Extended eMNIST
- For Paper 2: eMNIST, Coil-20, Olivetti faces
- For Paper 3: Mir flicker

## Acknowledgements
We extend our sincere gratitude to
- Pranav K Nayak (Teaching Assistant, UMC 203) for his support throughout the project
- Professor Chiranjib Bhattacharya and Professor N.Y.K. Shishir for providing the opportunity to explore this topic through a graded term paper in their course
