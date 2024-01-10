import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('formresponden.csv', delimiter=';', on_bad_lines='skip')

# Select only the relevant columns for clustering
relevant_columns = ['jam_dalam_seminggu', 'X_Gadget', 'Y_Fomo']
data_clustering = df[relevant_columns].dropna()

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on your data and then transform it
scaled_data = scaler.fit_transform(data_clustering)

# Train the KMeans model with the number of clusters you want, e.g., 2 for FOMO and not FOMO
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_data)

# Use PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Plot the reduced data points
plt.figure(figsize=(10, 7))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')

# Plot the centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=3, color='r')

# Title and labels
plt.title('2D PCA of KMeans Clusters')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')

# Save the plot as a file
plt.savefig('static/kmeans_clusters.png')

# Save the scaler and model for later use
dump(scaler, 'scaler.joblib')
dump(kmeans, 'kmeans_model.joblib')

# Return the path to the saved plot
'kmeans_clusters.png'
