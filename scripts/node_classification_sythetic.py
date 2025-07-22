import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic data
np.random.seed(42)

# ID class 0 and 1: Gaussian blobs
n_id = 100
X_id_0 = np.random.normal(loc=[-2, 0], scale=0.5, size=(n_id, 2))
X_id_1 = np.random.normal(loc=[2, 0], scale=0.5, size=(n_id, 2))
X_id = np.vstack([X_id_0, X_id_1])
y_id = np.array([0]*n_id + [1]*n_id)

# OOD samples: centered near the ID decision boundary
n_ood = 100
X_ood = np.random.normal(loc=[0, 0], scale=1.5, size=(n_ood, 2))
y_ood = np.array([-1]*n_ood)  # OOD label

# Step 2: Create two representations: one compressed (IB-like), one raw
# IB-like: project to 1D (simulate compression)
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
Z_id_compressed = pca.fit_transform(X_id)
Z_ood_compressed = pca.transform(X_ood)

# Raw: no compression
Z_id_raw = X_id.copy()
Z_ood_raw = X_ood.copy()

# Step 3: Train classifiers and evaluate
def train_and_eval(Z_id, Z_ood, compress_label):
    X_train, X_test, y_train, y_test = train_test_split(Z_id, y_id, test_size=0.3, random_state=42)
    clf = LogisticRegression().fit(X_train, y_train)
    
    # Decision boundary visualization
    grid_x = np.linspace(-4, 4, 200)
    if Z_id.shape[1] == 1:
        grid = grid_x.reshape(-1, 1)
    else:
        grid_y = np.linspace(-4, 4, 200)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid = np.c_[xx.ravel(), yy.ravel()]
    
    probs = clf.predict_proba(grid)[:, 1].reshape(-1 if Z_id.shape[1] == 1 else (200, 200))
    
    # OOD detection: use max softmax prob as OOD score
    ood_scores = clf.predict_proba(Z_ood)[:, 1]
    id_scores = clf.predict_proba(Z_id)[:, 1]
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)

    return clf, grid_x, probs, auroc

clf_raw, gx_raw, probs_raw, auroc_raw = train_and_eval(Z_id_raw, Z_ood_raw, "Raw")
clf_cmp, gx_cmp, probs_cmp, auroc_cmp = train_and_eval(Z_id_compressed, Z_ood_compressed, "Compressed")

# Plotting
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw
axes[0].scatter(Z_id_raw[:, 0], Z_id_raw[:, 1], c=y_id, cmap='coolwarm', alpha=0.6, label='ID')
axes[0].scatter(Z_ood_raw[:, 0], Z_ood_raw[:, 1], c='green', marker='x', label='OOD')
axes[0].contour(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200), probs_raw.reshape(200, 200), levels=[0.5], colors='black')
axes[0].set_title(f"Raw features\nOOD AUROC = {auroc_raw:.3f}")
axes[0].legend()

# Compressed
axes[1].scatter(Z_id_compressed[:n_id], np.zeros_like(Z_id_compressed[:n_id]), c='blue', label='ID-0')
axes[1].scatter(Z_id_compressed[n_id:], np.zeros_like(Z_id_compressed[n_id:]), c='red', label='ID-1')
axes[1].scatter(Z_ood_compressed, np.zeros_like(Z_ood_compressed), c='green', marker='x', label='OOD')
axes[1].axvline(x=clf_cmp.intercept_/-clf_cmp.coef_[0], color='black', linestyle='--')
axes[1].set_ylim(-1, 1)
axes[1].set_title(f"Compressed (1D) features\nOOD AUROC = {auroc_cmp:.3f}")
axes[1].legend()

plt.tight_layout()
plt.show()
