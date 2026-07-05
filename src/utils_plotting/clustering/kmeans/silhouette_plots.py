def plot_silhouette_profiles(X_subsample, k_candidates=[2, 3, 4, 5]):
    """
    Generates high-level Yellowbrick Silhouette visualizer profiles across 
    the selected K candidates using the memory-safe spatial subsample.
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import silhouette_visualizer

    # Iterate through each K and trigger the yellowbrick profile display
    for k_candidate in k_candidates:
        plt.figure(figsize=(8, 5))

        # Run the high-level yellowbrick visualizer directly on the subsample
        silhouette_visualizer(
            KMeans(n_clusters=k_candidate, init='k-means++', n_init=5, max_iter=150, random_state=42),
            X_subsample,
            colors='yellowbrick'
        )

def plot_definitive_silhouette(X_subsample, n_clusters=4):
    """
    Plots the definitive Yellowbrick Silhouette profile for the selected 
    target cluster count using the memory-safe spatial subsample.
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import silhouette_visualizer

    plt.figure(figsize=(8, 5))
    silhouette_visualizer(
        KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=1000, random_state=42),
        X_subsample,
        colors='yellowbrick'
    )