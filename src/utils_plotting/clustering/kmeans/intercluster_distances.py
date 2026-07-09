import matplotlib.pyplot as plt

def plot_intercluster_distances(X_final_preprocessed, n_clusters=4):
    """
    Generates a professional Intercluster Distance map via Yellowbrick to visualize
    the relative embedding separation and size weights of the 4 definitive groups.
    """
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import InterclusterDistance

    plt.figure(figsize=(9, 6))

    # Instantiate the visualizer with our optimal K settings
    visualizer = InterclusterDistance(
        KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=1000, random_state=42),
        colors='yellowbrick',
        random_state = 42
    )

    # Fit directly on the full preprocessed matrix
    visualizer.fit(X_final_preprocessed)
    visualizer.show()