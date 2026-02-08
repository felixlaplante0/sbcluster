import matplotlib.pyplot as plt
import numpy as np
from sbcluster import SpectralBridges
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.neighbors import KNeighborsRegressor


# Set random seed
np.random.seed(0)

# Load MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
X = PCA(n_components=32, random_state=42).fit_transform(mnist.data)
y = mnist.target

# Run SpectralBridges as a dimensionality reduction method
model = SpectralBridges(n_clusters=10, n_nodes=500, random_state=42)
model.fit(X)

# Spectral embedding
embedding = (s := SpectralEmbedding(n_components=32)).fit_transform(
    model.affinity_matrix_
)
embedding = TSNE(n_components=2).fit_transform(
    embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
)
knn = KNeighborsRegressor(n_neighbors=1).fit(model.cluster_centers_, embedding)
y_pred = knn.predict(X)

# Visualize
plt.scatter(y_pred[:, 0], y_pred[:, 1], c=y.astype(int), s=1)
plt.title("Spectral embedding from affinity matrix")
plt.show()
