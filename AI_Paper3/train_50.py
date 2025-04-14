import numpy as np

# Load original training indices
train_idx = np.load('data/mirflickr/mirflickr_train_idx.npy')

# Randomly sample 50% of the training indices without replacement
np.random.seed(42)  # For reproducibility
subset_train_idx = np.random.choice(train_idx, size=int(len(train_idx)*0.5), replace=False)

# Save the subset indices
np.save('data/mirflickr/mirflickr_train_50pct_idx.npy', subset_train_idx)