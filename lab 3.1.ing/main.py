import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kmeans(data, K, max_iters=100):

    n_samples, n_features = data.shape

    # Initialize centroids randomly
    # We choose K points from the data as initial centroids
    idx = np.random.choice(n_samples, K, replace=False)
    centroids = data[idx, :]

    labels = np.zeros(n_samples, dtype=int)


    for _ in range(max_iters):  # Stopping criterion: Max iterations

        # Calculate distance and assign points to the nearest cluster
        for i in range(n_samples):
            point = data[i, :]
            # Calculate Euclidean distance from 'point' to all 'centroids'
            distances = np.linalg.norm(point - centroids, axis=1)
            # Assign the label of the nearest cluster
            labels[i] = np.argmin(distances)

        # Save old centroids for convergence check
        old_centroids = np.copy(centroids)

        # Update centroids
        # (by calculating the mean of the points in the cluster)
        for k in range(K):
            # Find all points assigned to cluster k
            points_in_cluster = data[labels == k]

            # If the cluster is not empty, calculate the new mean
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)

        # Check for convergence
        if np.allclose(old_centroids, centroids):
            break

    # Calculate the number of items for the report
    counts = np.zeros(K, dtype=int)
    for k in range(K):
        counts[k] = np.sum(labels == k)

    return centroids, counts

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('WINE.txt', header=None, sep=r'\s+')

    # Extract the 13 features (columns 1 to 13)
    X = data.iloc[:, 1:].values

    # Calculate the mean and standard deviation for each feature (column)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    # Standardize the data
    X_scaled = (X - X_mean) / X_std

    # Note: np.cov expects features as rows, so we transpose X_scaled
    # calculate covariance matrix
    Q = np.cov(X_scaled.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(Q)

    # Sort the components based on the eigenvalue (variance)
    idx = np.argsort(eigenvalues)[::-1]  # Indices for sorting in descending order
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Each column is an eigenvector (a principal component),
    principal_component_matrix = sorted_eigenvectors

    # Project the standardized data onto the new eigenvector basis
    # X_new = X_scaled * W (where W is the eigenvector matrix)
    # the lines are the projection of each feature onto the new direction of the eigenvalues
    X_pca = X_scaled.dot(principal_component_matrix)

    # X_pca contains the "new feature values" requested
    print("Original data shape:", X_scaled.shape)
    print("Transformed data shape (PCA):", X_pca.shape)

    # Extract the first two principal components
    PC1 = X_pca[:, 0]
    PC2 = X_pca[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(PC1, PC2)
    plt.title('WINE Data Visualization - Top 2 Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plt.savefig('pca_top2_visualization.png')

    # start experiments

    K = 3

    # Experiment 1: Top 2 Components
    print("Started first experiment")
    X_top2 = X_pca[:, :2]
    centroids_2, counts_2 = kmeans(X_top2, K)
    print("Finished first experiment")

    # Experiment 2: Top 5 Components
    print("Started second experiment")
    X_top5 = X_pca[:, :5]
    centroids_5, counts_5 = kmeans(X_top5, K)
    print("Finished second experiment")

    # Experiment 3: Top 8 Components
    print("Started third experiment")
    X_top8 = X_pca[:, :8]
    centroids_8, counts_8 = kmeans(X_top8, K)
    print("Finished third experiment")

    # --- Print results for the report table ---

    print("\n--- K-means Results (K=3) ---")

    print("\n--- Experiment 1: Top 2 Components ---")
    for k in range(K):
        print(f"Cluster {k + 1}: N. Items = {counts_2[k]}, Centroid = {centroids_2[k]}")

    print("\n--- Experiment 2: Top 5 Components ---")
    for k in range(K):
        print(f"Cluster {k + 1}: N. Items = {counts_5[k]}, Centroid = {centroids_5[k]}")

    print("\n--- Experiment 3: Top 8 Components ---")
    for k in range(K):
        print(f"Cluster {k + 1}: N. Items = {counts_8[k]}, Centroid = {centroids_8[k]}")


    plt.show()
