import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def plot_heat_by_clus(biomass_data_clustered,products):
    '''
    Plots heatmap of product correlations based on how products have clustered

    Input: 
    biomass_data_clustered - saved dataframe of original data with clustering results appended
    products - array of all product names
    '''
    if biomass_data_clustered.shape[0] > biomass_data_clustered.shape[1]:
        print("biomass_data_clustered has more rows than columns, clustering occured on locations")
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(biomass_data_clustered)

        # Create a DataFrame for the heatmap data
        heatmap_data = pd.DataFrame(index=products, columns=df['cluster'].unique())

        # Group data by cluster
        grouped = df.groupby('cluster')

        # Create subplots for each cluster
        fig, axes = plt.subplots(len(grouped), 1, figsize=(10, 6 * len(grouped)))

        # Calculate correlations and plot heatmaps for each cluster
        for i, (cluster, data) in enumerate(grouped):
            corr_matrix = data[products].corr()
            sns.heatmap(corr_matrix, ax=axes[i], cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
            axes[i].set_title(f'Correlation Matrix for Cluster {int(cluster)}')
            axes[i].set_xlabel('Products')
            axes[i].set_ylabel('Products')

        plt.tight_layout()
        plt.show()
        return None
    elif biomass_data_clustered.shape[0] < biomass_data_clustered.shape[1]:
        print("biomass_data_clustered has more columns than rows, clustering occured on products")
        # Create a correlation matrix for each cluster
        correlation_matrices = {}
        for cluster_id in biomass_data_clustered['cluster'].unique():
            cluster_data = biomass_data_clustered[biomass_data_clustered['cluster'] == cluster_id]
            cluster_data_products = cluster_data[cluster_data['index'].isin(products)]

            # Check if the cluster contains more than one product
            if len(cluster_data_products) > 1:
                print(f"Cluster {cluster_id} contains multiple products, plotting heatmap...")
                correlation_matrix = cluster_data_products.set_index('index').T.corr()
                correlation_matrices[cluster_id] = correlation_matrix

        # Plot the heatmap for each cluster
        for cluster_id, correlation_matrix in correlation_matrices.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            plt.title(f'Correlation Matrix for Cluster {cluster_id}')
            plt.xlabel('Product')
            plt.ylabel('Product')
            plt.tight_layout()
            plt.show()
        return correlation_matrices
    else:
        print("biomass_data_clustered has an equal number of rows and columns. Unclear how clustering was done, please debug")
        return none

    
def print_indices(data_scaled, predicted_labels):
    '''
    Evaluate clustering performance using:
    Silhouette Coefficient-
    The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    Scores around zero indicate overlapping clusters.
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster
    Calinski-Harabasz Index-
    Variance Ratio Criterion - can be used to evaluate the model, 
    where a higher Calinski-Harabasz score relates to a model with better defined clusters
    Davies-Bouldin Index-
    This index signifies the average ‘similarity’ between clusters,
    where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
    Zero is the lowest possible score. Values closer to zero indicate a better partition.
    '''
    silhouette_score_value = silhouette_score(data_scaled, predicted_labels)
    calinski_harabasz_score_value = calinski_harabasz_score(data_scaled, predicted_labels)
    davies_bouldin_score_value = davies_bouldin_score(data_scaled, predicted_labels)
    
    print(f"Silhouette Coefficient: {silhouette_score_value}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score_value}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score_value}")
    
    
    
def generate_noisy_data(original_data, n_vectors, scale_factor):
    """
    Generate n_vectors of noisy data from the given original_data using a scaling factor.
    
    Args:
        original_data (numpy.ndarray): The original data array.
        n_vectors (int): The number of noisy data vectors to generate.
        scale_factor (float): The scaling factor for the standard deviation of the noise.
        
    Returns:
        numpy.ndarray: A 2D array containing the n_vectors of noisy data,
                        with each row representing one noisy data vector.
    """
    mean = np.mean(original_data)
    std_dev = np.std(original_data)
    
    noisy_data_list = []
    for i in range(n_vectors):
        scaled_std_dev = std_dev * (1 + i * scale_factor)
        noise = np.random.normal(mean, scaled_std_dev, original_data.shape)
        noisy_data = original_data + noise
        noisy_data_list.append(noisy_data)
    
    return np.vstack(noisy_data_list)

# def generate_noisy_data(original_data, n_vectors):
#     """
#     Generate n_vectors of noisy data from the given original_data.
    
#     Args:
#         original_data (numpy.ndarray): The original data array.
#         n_vectors (int): The number of noisy data vectors to generate.
        
#     Returns:
#         numpy.ndarray: A 2D array containing the n_vectors of noisy data,
#                         with each row representing one noisy data vector.
#     """
#     mean = np.mean(original_data)
#     std_dev = np.std(original_data)
    
#     noisy_data_list = []
#     for _ in range(n_vectors):
#         noise = np.random.normal(mean, std_dev, original_data.shape)
#         noisy_data = original_data + noise
#         noisy_data_list.append(noisy_data)
    
#     return np.vstack(noisy_data_list)

def get_dist(data, labels):
    """
    Inputs
        data: input data for clustering
        labels: labels of clustering output
    
    Find cluster centers, distances between clusters and distances between data points and centers 
    Use this to complete calculations AFTER clustering is complete
    This is based on euclidean metric, NOT the optimization methods or representation sample of the given algorithm
    
    Outputs:
        cluster_centers: center of each cluster. expect this as a 1xn array, where n is the number of 
                columns in the data set used
        cluster_distances: distance from the cluster of each cluster to another one. expect this as an cxc matrix,
                where c is the number of clusters. the diagonal is always 0 (distance to itself)
        data_cluster_distances: distance from each sample to the center of each cluster. Expect this as a mxc matrix,
                where m is number of samples and c is the number of clusters
    """

    # Get the number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Calculate cluster centers
    if -1 in labels:
        cluster = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])
        outlier = np.array([data[labels == -1].mean(axis=0) ])
        cluster_centers = np.vstack((cluster, outlier))
    else:
        cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])

#     print("Cluster Centers:")
#     print(cluster_centers)

    # Compute the pairwise distances between cluster centers
    cluster_distances = cdist(cluster_centers, cluster_centers, metric='euclidean')
    # Print the distances between cluster centers
    print("Distances between cluster centers:")
    print(cluster_distances)

    # Compute the distances between each data point and cluster centers
    data_cluster_distances = cdist(data, cluster_centers, metric='euclidean')

    # Print the distances between each data point and cluster centers
    print("\nDistances between data points and cluster centers:")
    for i, distances in enumerate(data_cluster_distances):
        print(f"Data point {i}: {distances}")
        
    return 

# # Calculate cluster centers for AgglomerativeClustering
# cluster_centers = np.array([data_scaled[labels == i].mean(axis=0) for i in range(agg.n_clusters_)])

# print("\nAgglomerativeClustering Cluster Centers:")
# print(cluster_centers)

# # Compute the pairwise distances between cluster centers
# cluster_distances = cdist(cluster_centers, cluster_centers, metric='euclidean')
# # Print the distances between cluster centers
# print("Distances between cluster centers:")
# print(cluster_distances)

# # Compute the distances between each data point and cluster centers
# data_cluster_distances = cdist(data_scaled, cluster_centers, metric='euclidean')

# # Print the distances between each data point and cluster centers
# print("\nDistances between data points and cluster centers:")
# for i, distances in enumerate(data_cluster_distances):
#     print(f"Data point {i}: {distances}")