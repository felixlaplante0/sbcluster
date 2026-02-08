Spectral Bridges
================

**sbcluster** is a Python package that implements a novel clustering algorithm combining k-means and spectral clustering techniques, called **Spectral Bridges**. It leverages efficient affinity matrix computation and merges clusters based on a connectivity measure inspired by SVM's margin concept. This package is designed to provide robust clustering solutions, particularly suited for large datasets.

Features
--------

- **Spectral Bridges Clustering Algorithm**: Integrates k-means and spectral clustering with efficient affinity matrix calculation for improved clustering results.
- **Scalability**: Designed to handle large datasets by optimizing cluster formation through advanced affinity matrix computations.
- **Customizable**: Parameters such as number of clusters, iterations, and random state allow flexibility in clustering configurations.
- **Model selection**: Automatic model selection for number of nodes (m) according to a normalized eigengap metric.
- **scikit-learn**: Native integration with the standard API, with easy options for model selection and evaluation.

Speed
-----

Spectral Bridges utilizes fastkmeanspp's efficient implementation for KMeans, which makes it remarkably fast even with large scale datasets.

Installation
------------

You can install the package via pip:

.. code-block:: bash

   pip install sbcluster

Usage
-----

Example:

.. code-block:: python

   from time import time

   import matplotlib.pyplot as plt
   import numpy as np
   from sbcluster import SpectralBridges, ngap_scorer
   from sklearn.cluster import SpectralClustering
   from sklearn.metrics import adjusted_rand_score
   from sklearn.model_selection import GridSearchCV

   # Load some synthetic data
   data = np.genfromtxt("datasets/impossible.csv", delimiter=",")
   X, y = data[:, :-1], data[:, -1]

   # Define the parameter grid
   param_grid = {"n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
   cv = [(np.arange(X.shape[0]), np.arange(X.shape[0]))] * 5

   # Perform grid search for optimal parameters
   grid_search = GridSearchCV(
      estimator=SpectralBridges(n_clusters=2, n_nodes=250),
      param_grid=param_grid,
      scoring=ngap_scorer,
      cv=cv,
      verbose=1,
   )

   # Fit the grid search
   grid_search.fit(X)

   # Print the results
   print(grid_search.cv_results_["mean_test_score"])
   print(grid_search.best_params_)

   # Make predictions with the best model
   guess = grid_search.best_estimator_.predict(X)
   ari = adjusted_rand_score(y, guess)

   # Print the ARI
   print(f"Adjusted Rand Index: {ari}")

   # Visualize the clustering results
   plt.scatter(X[:, 0], X[:, 1], c=guess, alpha=0.1)
   plt.scatter(
      grid_search.best_estimator_.cluster_centers_[:, 0],
      grid_search.best_estimator_.cluster_centers_[:, 1],
      c=grid_search.best_estimator_.cluster_labels_,
      marker="X",
   )
   plt.title("Clustered data and centroids with best SpectralBridges fit")
   plt.show()

   # Compare with sklearn's SpectralClustering
   sc_low = SpectralClustering(n_clusters=7).fit(X)

   plt.scatter(X[:, 0], X[:, 1], c=sc_low.labels_)
   plt.title("Spectral Clustering of the original dataset, gamma=1.0")
   plt.show()

   sc_high = SpectralClustering(n_clusters=7, gamma=5).fit(X)

   plt.scatter(X[:, 0], X[:, 1], c=sc_high.labels_)
   plt.title("Spectral Clustering of the original dataset, gmma=5.0")
   plt.show()

   # Comapre times
   start = time()
   grid_search.best_estimator_.fit(X)
   end = time()
   print("SpectralBridges fit time:", end - start)

   start = time()
   sc_low.fit(X)
   end = time()
   print("SpectralClustering fit time:", end - start)

API Reference
-------------

.. autoclass:: sbcluster._bridges.SpectralBridges
   :members:
   :undoc-members:
   :show-inheritance:
