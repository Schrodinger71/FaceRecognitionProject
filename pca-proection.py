import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================
# LOAD EMBEDDINGS
# =========================
with open("embeddings.pkl", "rb") as f:
    X, y = pickle.load(f)

X = np.asarray(X)
y = np.asarray(y)


# =========================
# PCA: 128D -> 2D
# =========================
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)


# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 6))

for label in np.unique(y):
    mask = y == label
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        label=f"person_{label}",
        alpha=0.7
    )

plt.title("PCA projection of face embeddings")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.legend()
plt.grid(True)

plt.show()
