import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#Function to group my hamming distance and similiarity threshold -------------------------------------------------

def group_samples_by_similarity_threshold(matrix, weights, similarity_threshold, columns, index):
    # Convert matrix to numpy array if it's not already
    matrix = np.array(matrix)
    weights = np.array(weights)
    
    # Calculate weighted Hamming similarity
    similarity_matrix = weighted_hamming_similarity(matrix, weights)
    
    # Create a DataFrame with the original data
    df = pd.DataFrame(matrix, columns=feature_cols, index=products)
    
    # Initialize groups
    groups = [-1] * len(matrix)
    current_group = 0
    
    for i in range(len(matrix)):
        if groups[i] == -1:
            # Start a new group
            groups[i] = current_group
            # Find all samples similar enough to be in this group
            similar_samples = np.where(similarity_matrix[i] >= similarity_threshold)[0]
            for j in similar_samples:
                if groups[j] == -1:
                    groups[j] = current_group
            current_group += 1
    
    df['group'] = groups
    
    return df

# def weighted_hamming_similarity(matrix, weights):
#     def weighted_hamming(u, v):
#         return np.sum(weights * (u == v)) / np.sum(weights)
    
#     distances = pdist(matrix, metric=lambda u, v: 1 - weighted_hamming(u, v))
#     return 1 - squareform(distances)


def weighted_hamming(u, v, weights):
    return np.sum(weights * (u == v)) / np.sum(weights)

def weighted_hamming_similarity(matrix, weights):
    distances = pdist(matrix, metric=lambda u, v: 1 - weighted_hamming(u, v, weights))
    return 1 - squareform(distances)

def analyze_products(matrix, weights, similarity_threshold=None, columns=None, index=None):
    # Convert matrix to numpy array if it's not already
    matrix = np.array(matrix)
    weights = np.array(weights)
    
    # Calculate weighted Hamming distances
    distance_matrix = weighted_hamming_similarity(matrix, weights)
    
    # Create a DataFrame with the original data
    df = pd.DataFrame(matrix, columns=columns, index=index)
    
    # Create a DataFrame for the distance matrix
    distance_df = pd.DataFrame(distance_matrix, columns=index, index=index)
    
    print("Weighted Hamming Distances:")
    print(distance_df)
    
    if similarity_threshold is not None:
        # Initialize groups
        groups = [-1] * len(matrix)
        current_group = 0
        
        for i in range(len(matrix)):
            if groups[i] == -1:
                # Start a new group
                groups[i] = current_group
                # Find all samples similar enough to be in this group
                similar_samples = np.where(distance_matrix[i] >= similarity_threshold)[0]
                for j in similar_samples:
                    if groups[j] == -1:
                        groups[j] = current_group
                current_group += 1
        
        df['group'] = groups
        
        # Print out common variables for each group
        for group in range(current_group):
            group_samples = df[df['group'] == group]
            common_vars = group_samples.iloc[:, :-1].apply(lambda x: len(set(x)) == 1)
            common_var_names = common_vars[common_vars].index.tolist()
            
            print(f"\nGroup {group}:")
            print(f"Samples: {', '.join(group_samples.index)}")
            print(f"Common variables: {', '.join(common_var_names)}")
        
        return df, distance_df
    else:
        return distance_df

#functions to help find appropriate threshold --------------------------------------------

def elbow_method(matrix, weights, thresholds, feature_cols, products):
    num_groups = []
    for threshold in thresholds:
        result = group_samples_by_similarity_threshold(matrix, weights, threshold, feature_cols, products)
        num_groups.append(result['group'].nunique())
    
    plt.plot(thresholds, num_groups, 'bo-')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Number of Groups')
    plt.title('Elbow Method')
    plt.show()

def silhouette_analysis(matrix, weights, thresholds):
    scores = []
    for threshold in thresholds:
        result = group_samples_by_similarity_threshold(matrix, weights, threshold, feature_cols, products)
        unique_groups = result['group'].nunique()

        if 1 < unique_groups < len(matrix):
            try:
                score = silhouette_score(matrix, result['group'])
                scores.append(score)
            except ValueError as e:
                print(f"Error at threshold {threshold}: {e}")
                scores.append(0)
        else:
            print(f"Invalid number of clusters ({unique_groups}) at threshold {threshold}. Skipping.")
            scores.append(0)
    
    plt.plot(thresholds, scores, 'bo-')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.show()

def hierarchical_clustering(matrix, weights):
    similarity_matrix = weighted_hamming_similarity(matrix, weights)
    linkage_matrix = linkage(1 - similarity_matrix, method='average')
    
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Products')
    plt.ylabel('Distance')
    plt.show()

def dbscan_clustering(matrix, weights):
    similarity_matrix = weighted_hamming_similarity(matrix, weights)
    distances = 1 - similarity_matrix
    
    dbscan = DBSCAN(metric='precomputed', eps=0.5)  # Adjust eps as needed
    clusters = dbscan.fit_predict(distances)
    
    return clusters

#visualizations --------------------------------------------------------------------------

# Use this function after you've calculated the distances
def plot_product_distances(distance_df):
    # 1. Multidimensional Scaling (MDS)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(distance_df)

    plt.figure(figsize=(10, 8))
    plt.scatter(mds_coords[:, 0], mds_coords[:, 1], marker='o')
    for i, product in enumerate(distance_df.index):
        plt.annotate(product, (mds_coords[i, 0], mds_coords[i, 1]), xytext=(5, 5), 
                     textcoords='offset points')
    plt.title('Product Distances - Multidimensional Scaling')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()

    # 2. Heatmap with Hierarchical Clustering
    plt.figure(figsize=(12, 10))
    sns.clustermap(distance_df, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Product Distances - Heatmap with Hierarchical Clustering')
    plt.tight_layout()
    plt.show()

# Use this function after you've prepared your matrix
def perform_pca_analysis(matrix, feature_cols, products):
    # Convert categorical variables to dummy variables
    df = pd.DataFrame(matrix, columns=feature_cols, index=products)
    df_encoded = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'])
    
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_encoded)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.show()
    
    # Plot first two principal components
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    for i, product in enumerate(products):
        plt.annotate(product, (pca_result[i, 0], pca_result[i, 1]), xytext=(5, 5), 
                     textcoords='offset points')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Products')
    plt.tight_layout()
    plt.show()
    
    # Return PCA results, the PCA object, and the encoded DataFrame
    return pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])], index=products), pca, df_encoded

def plot_loading_vectors(pca, feature_names):
    plt.figure(figsize=(12, 10))
    
    # Get the loading vectors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create the scatter plot of loadings
    plt.scatter(loadings[:, 0], loadings[:, 1])
    
    # Add arrows
    for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
        plt.arrow(0, 0, x, y, color='r', alpha=0.5)
        plt.text(x, y, feature_names[i], fontsize=9)
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
    plt.title("PCA Loading Plot")
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()