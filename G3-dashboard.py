import os
import re
import nltk
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import scipy.sparse as sp
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from gensim.models import LdaModel
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import defaultdict
from gensim.corpora import Dictionary
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from gensim.models import CoherenceModel
from gensim.matutils import Sparse2Corpus
from gensim.models.ldamodel import LdaModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import silhouette_score as sil_score

transformed_data = pd.read_csv('transformed_data.csv')

num_clusters_range = range(2, 10)

#Hierarchical Clustering
# Compute Euclidean distances between data points in transformed_data
euclidean_dist = euclidean_distances(transformed_data)

# Print the shape of the resulting distance matrix
print("Shape of Euclidean distance matrix:", euclidean_dist.shape)

# Initialize hierarchical clustering with single linkage
single_clustering = AgglomerativeClustering(n_clusters=10, linkage='single')

# Fit the clustering model to the data
single_clusters = single_clustering.fit_predict(euclidean_dist)

# Initialize hierarchical clustering with complete linkage
complete_clustering = AgglomerativeClustering(n_clusters=10, linkage='complete')

# Fit the clustering model to the data
complete_clusters = complete_clustering.fit_predict(euclidean_dist)

# Initialize hierarchical clustering with average linkage
average_clustering = AgglomerativeClustering(n_clusters=10, linkage='average')

# Fit the clustering model to the data
average_clusters = average_clustering.fit_predict(euclidean_dist)

# Initialize hierarchical clustering with Ward's linkage
ward_clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')

# Fit the clustering model to the data
ward_clusters = ward_clustering.fit_predict(euclidean_dist)

# Initialize hierarchical clustering with the chosen linkage criterion
clustering = AgglomerativeClustering(n_clusters=10, linkage='average')  # Adjust linkage as needed

# Fit the clustering model to the data
clusters = clustering.fit_predict(euclidean_dist)

linkage_matrix = linkage(transformed_data, method='ward', metric='euclidean')

# NMF Clustering
transformed_data_non_negative = np.maximum(transformed_data, 0)

# Define a range of cluster numbers to evaluate
cluster_range = range(2, 11)

# Spectral Clustering
gamma = 1.0  # Adjust the value of gamma as needed
similarity_matrix = rbf_kernel(transformed_data, gamma=gamma)

# Perform Spectral Clustering
clustering = SpectralClustering(n_clusters=10, affinity='precomputed')
cluster_labels = clustering.fit_predict(similarity_matrix)






# Define functions to generate Silhouette Score plots

def plot_kmeans_silhouette_scores(num_clusters):
    silhouette_scores = []
    for num_clusters in range(2, num_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(transformed_data)
        silhouette_avg = sil_score(transformed_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, num_clusters + 1), silhouette_scores, marker='o', linestyle='-')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for Different Numbers of Clusters (K-Means Clustering)')
    ax.grid(True)
    ax.set_facecolor('black')  # Set background color to black
    ax.spines['bottom'].set_color('white')  # Set bottom spine color to white
    ax.spines['left'].set_color('white')  # Set left spine color to white
    ax.xaxis.label.set_color('white')  # Set x-axis label color to white
    ax.yaxis.label.set_color('white')  # Set y-axis label color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.title.set_color('white')

    st.pyplot(fig, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    return fig

def plot_hierarchical_silhouette_scores(num_clusters):
    silhouette_scores = []
    for n_clusters in range(2, num_clusters + 1): 
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(linkage_matrix)
        silhouette_avg = sil_score(linkage_matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, num_clusters + 1), silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.grid(True)
    ax.set_title('Silhouette Score for Different Numbers of Clusters (Hierarchical Clustering)')
    ax.set_facecolor('black')  # Set background color to black
    ax.spines['bottom'].set_color('white')  # Set bottom spine color to white
    ax.spines['left'].set_color('white')  # Set left spine color to white
    ax.xaxis.label.set_color('white')  # Set x-axis label color to white
    ax.yaxis.label.set_color('white')  # Set y-axis label color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.title.set_color('white')

    st.pyplot(fig, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    return fig

def plot_nmf_silhouette_scores(num_clusters):
    silhouette_scores = []
    for n_clusters in range(2, num_clusters + 1):
        nmf = NMF(n_components=n_clusters, random_state=42)
        nmf_representation = nmf.fit_transform(transformed_data_non_negative)
        cluster_labels = np.argmax(nmf_representation, axis=1)
        silhouette_avg = sil_score(transformed_data_non_negative, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, num_clusters + 1), silhouette_scores, marker='o', label='Silhouette Score')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.set_title('Silhouette Score for Different Numbers of Clusters (NMF Clustering)')
    ax.legend()
    ax.set_facecolor('black')  # Set background color to black
    ax.spines['bottom'].set_color('white')  # Set bottom spine color to white
    ax.spines['left'].set_color('white')  # Set left spine color to white
    ax.xaxis.label.set_color('white')  # Set x-axis label color to white
    ax.yaxis.label.set_color('white')  # Set y-axis label color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.title.set_color('white')

    st.pyplot(fig, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    return fig

def plot_spectral_silhouette_scores(num_clusters):
    silhouette_scores = []
    for n_clusters in range(2, num_clusters + 1):
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        cluster_labels = spectral_clustering.fit_predict(similarity_matrix)
        silhouette_avg = sil_score(transformed_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, num_clusters + 1), silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.grid(True)
    ax.set_title('Silhouette Score for Different Numbers of Clusters (Spectral Clustering)')
    ax.set_facecolor('black')  # Set background color to black
    ax.spines['bottom'].set_color('white')  # Set bottom spine color to white
    ax.spines['left'].set_color('white')  # Set left spine color to white
    ax.xaxis.label.set_color('white')  # Set x-axis label color to white
    ax.yaxis.label.set_color('white')  # Set y-axis label color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.title.set_color('white')

    st.pyplot(fig, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    return fig

# Custom background and color theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Font and text style
st.markdown(
    """
    <style>
    /* CSS to style the title */
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 50px;
        font-weight: bold;
        color: #33333;
    }

    /* CSS to style the sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }

    /* CSS to style the slider label and value */
    .slider-container .label {
        font-size: 66px !important; /* Set font size for the slider label */
        color: #ffffff !important; /* Set font color for the slider label */
    }
    .slider-container .slider {
        font-size: 46px !important; /* Set font size for the slider value */
        color: #ffffff !important; /* Set font color for the slider value */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def set_plot_style():
    # Set background color to black and font color to white
    plt.style.use('dark_background')
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
# Function to visualize KMeans clusters
def visualize_kmeans_clusters(transformed_data, n_clusters):
    set_plot_style()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(transformed_data)
    
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d_transformed_data = pca_2d.fit_transform(transformed_data)

    plt.figure(figsize=(8, 5))
    for i in range(n_clusters):
        plt.scatter(pca_2d_transformed_data[cluster_labels == i, 0], 
                    pca_2d_transformed_data[cluster_labels == i, 1], 
                    label=f'Cluster {i}', alpha=0.7)
    plt.title('Visualization of KMeans Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.legend(loc='upper right')
    st.pyplot(plt)

def visualize_hierarchical_clusters(transformed_data, n_clusters):
    set_plot_style()
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(transformed_data)
    
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d_transformed_data = pca_2d.fit_transform(transformed_data)

    plt.figure(figsize=(8, 5))
    for i in range(n_clusters):
        plt.scatter(pca_2d_transformed_data[cluster_labels == i, 0], 
                    pca_2d_transformed_data[cluster_labels == i, 1], 
                    label=f'Cluster {i}', alpha=0.7)
    plt.title('Visualization of Hierarchical Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.legend(loc='upper right')
    st.pyplot(plt)

def visualize_nmf_clusters(transformed_data, n_clusters):
    set_plot_style()
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d_transformed_data = pca_2d.fit_transform(transformed_data)

    plt.figure(figsize=(8, 5))
    for i in range(n_clusters):
        plt.scatter(pca_2d_transformed_data[cluster_labels == i, 0], 
                    pca_2d_transformed_data[cluster_labels == i, 1], 
                    label=f'Cluster {i}', alpha=0.7)
    plt.title('Visualization of NMF Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.legend(loc='upper right')  
    st.pyplot(plt)

def visualize_spectral_clusters(transformed_data, n_clusters):
    set_plot_style()
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    cluster_labels = spectral_clustering.fit_predict(transformed_data)
    
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d_transformed_data = pca_2d.fit_transform(transformed_data)

    plt.figure(figsize=(8, 5))
    for i in range(n_clusters):
        plt.scatter(pca_2d_transformed_data[cluster_labels == i, 0], 
                    pca_2d_transformed_data[cluster_labels == i, 1], 
                    label=f'Cluster {i}', alpha=0.7)
    plt.title('Visualization of Spectral Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.legend(loc='upper right')
    st.pyplot(plt)



# Function to visualize clusters using t-SNE for KMeans
def visualize_clusters_tSNE_KMeans(transformed_data, num_clusters):
    set_plot_style()
    # Check if transformed_data is sparse and convert to dense array if necessary
    if sp.issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    # Reduce dimensionality using t-SNE with adjusted parameters
    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, learning_rate=200, random_state=42)
    tsne_representation = tsne.fit_transform(transformed_data)

    # Apply KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(transformed_data)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=cluster_labels, cmap='coolwarm')
    plt.title('Clusters Visualization using t-SNE and KMeans')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.colorbar(label='Cluster')

    # Add cluster labels
    for cluster in range(len(np.unique(cluster_labels))):
        cluster_center = np.mean(tsne_representation[cluster_labels == cluster], axis=0)
        plt.text(cluster_center[0], cluster_center[1], f'Cluster {cluster}', fontsize=12, ha='center', va='center', color='white', weight='bold')

    st.pyplot(plt)

# Function to visualize clusters using t-SNE for Hierarchical Clustering
def visualize_clusters_tSNE_Hierarchical(transformed_data, num_clusters):
    set_plot_style()
    # Check if transformed_data is sparse and convert to dense array if necessary
    if sp.issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    # Reduce dimensionality using t-SNE with adjusted parameters
    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, learning_rate=200, random_state=42)
    tsne_representation = tsne.fit_transform(transformed_data)

    # Perform Hierarchical Clustering with adjusted parameters
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', affinity='euclidean')
    cluster_labels = clustering.fit_predict(transformed_data)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=cluster_labels, cmap='coolwarm')
    plt.title('Clusters Visualization using t-SNE and Hierarchical Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.colorbar(label='Cluster')

    # Add cluster labels
    for cluster in range(len(np.unique(cluster_labels))):
        cluster_center = np.mean(tsne_representation[cluster_labels == cluster], axis=0)
        plt.text(cluster_center[0], cluster_center[1], f'Cluster {cluster}', fontsize=12, ha='center', va='center', color='white', weight='bold')

    st.pyplot(plt)

def visualize_clusters_tSNE_NMF(transformed_data, num_clusters):
    set_plot_style()
    # Check if transformed_data is sparse and convert to dense array if necessary
    if sp.issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    # Preprocess the data to ensure non-negativity
    scaler = MinMaxScaler()
    transformed_data = scaler.fit_transform(transformed_data)

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_representation = tsne.fit_transform(transformed_data)

    # Perform NMF Clustering with adjusted parameters
    nmf = NMF(n_components=num_clusters)
    nmf_representation = nmf.fit_transform(transformed_data)

    # Assign cluster labels based on argmax of NMF representation
    cluster_labels = nmf_representation.argmax(axis=1)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=cluster_labels, cmap='coolwarm')
    plt.title('Clusters Visualization using t-SNE and NMF Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.colorbar(label='Cluster')

    # Add cluster labels
    for cluster in range(num_clusters):
        cluster_center = np.mean(tsne_representation[cluster_labels == cluster], axis=0)
        plt.text(cluster_center[0], cluster_center[1], f'Cluster {cluster}', fontsize=12, ha='center', va='center', color='white', weight='bold')

    st.pyplot(plt)

# Function to visualize clusters using t-SNE for Spectral Clustering
def visualize_clusters_tSNE_Spectral(transformed_data, num_clusters):
    set_plot_style()
    # Check if transformed_data is sparse and convert to dense array if necessary
    if sp.issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    # Reduce dimensionality using t-SNE with adjusted parameters
    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, learning_rate=200, random_state=42)
    tsne_representation = tsne.fit_transform(transformed_data)

    # Perform Spectral Clustering with adjusted parameters
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
    cluster_labels = spectral_clustering.fit_predict(transformed_data)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=cluster_labels, cmap='coolwarm')
    plt.title('Clusters Visualization using t-SNE and Spectral Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.colorbar(label='Cluster')

    # Add cluster labels
    for cluster in range(len(np.unique(cluster_labels))):
        cluster_center = np.mean(tsne_representation[cluster_labels == cluster], axis=0)
        plt.text(cluster_center[0], cluster_center[1], f'Cluster {cluster}', fontsize=12, ha='center', va='center', color='white', weight='bold')

    st.pyplot(plt)

def visualize_best_model(transformed_data, num_clusters):
    plot_hierarchical_silhouette_scores(num_clusters)
    visualize_hierarchical_clusters(transformed_data, num_clusters)
    visualize_clusters_tSNE_Hierarchical(transformed_data, num_clusters)

# Main function
def main():
    transformed_data = pd.read_csv('transformed_data.csv')

    st.header('Dashboard of Newsgroup Dataset')
    
    # Add navigation for different clustering algorithms
    page = st.radio("\nGo to", ('Visualization of Silhouette Score for Different Numbers of Clusters', 'Visualization Number of Clusters Using PCA', 'Visualization Number of Clusters Using t-SNE', 'Visualization of the Best Model: Hierarchical Clustering Model'))

    if page == 'Visualization of Silhouette Score for Different Numbers of Clusters':
        st.subheader('Visualization of Silhouette Score for Different Numbers of Clusters')
        # First slider to select the number of clusters for silhouette score visualization
        num_clusters_silhouette = st.slider('Select the number of clusters for silhouette score', 2, 10, 2, key='silhouette_slider')

        # Display Silhouette Score plot for KMeans clustering
        st.subheader('K-Means Clustering')
        kmeans_plot = plot_kmeans_silhouette_scores(num_clusters_silhouette)
        #st.pyplot(kmeans_plot)
    
        # Display Silhouette Score plot for Hierarchical clustering
        st.subheader('Hierarchical Clustering')
        hierarchical_plot = plot_hierarchical_silhouette_scores(num_clusters_silhouette)
        #st.pyplot(hierarchical_plot)
    
        # Display Silhouette Score plot for NMF clustering
        st.subheader('NMF Clustering')
        nmf_plot = plot_nmf_silhouette_scores(num_clusters_silhouette)
        #st.pyplot(nmf_plot)
    
        # Display Silhouette Score plot for Spectral clustering
        st.subheader('Spectral Clustering')
        spectral_plot = plot_spectral_silhouette_scores(num_clusters_silhouette)
        #st.pyplot(spectral_plot)

    elif page == 'Visualization Number of Clusters Using PCA':
        st.subheader('Visualization of Number of Clusters Using PCA')
        
        # Second slider to select the number of clusters for PCA visualization
        num_clusters_pca = st.slider('Select the number of clusters for PCA', 2, 10, 2, key='pca_slider')
                
        # Visualize KMeans Clusters
        st.subheader('K-Means Clustering')
        visualize_kmeans_clusters(transformed_data, num_clusters_pca)

        # Visualize Hierarchical Clusters
        st.subheader('Hierarchical Clustering')
        visualize_hierarchical_clusters(transformed_data, num_clusters_pca)
        
        # Visualize NMF Clusters
        st.subheader('NMF Clustering')
        visualize_nmf_clusters(transformed_data, num_clusters_pca)
        
        # Visualize Spectral Clusters
        st.subheader('Spectral Clustering')
        visualize_spectral_clusters(transformed_data, num_clusters_pca)

    elif page == 'Visualization Number of Clusters Using t-SNE':
        st.subheader('Visualization Number of Clusters Using t-SNE')

        # Third slider to select the number of clusters for t-SNE visualization
        num_clusters_tSNE = st.slider('Select the number of clusters for t-SNE', 2, 10, 2, key='t-SNE_slider')

        st.subheader('K-Means Clustering')
        visualize_clusters_tSNE_KMeans(transformed_data, num_clusters_tSNE)

        st.subheader('Hierarchical Clustering')
        visualize_clusters_tSNE_Hierarchical(transformed_data, num_clusters_tSNE)

        st.subheader('NMF Clustering')
        visualize_clusters_tSNE_NMF(transformed_data, num_clusters_tSNE)

        st.subheader('Spectral Clustering')
        visualize_clusters_tSNE_Spectral(transformed_data, num_clusters_tSNE)

    elif page == 'Visualization of the Best Model: Hierarchical Clustering Model':
        st.subheader('Visualization of the Best Model: Hierarchical Clustering Model')
        
        # Fourth slider to select the number of clusters for best model
        num_clusters_HC = st.slider('Select the number of clusters for best model', 2, 10, 2, key='hc_slider')
        st.write('After compared silhouette score between these algorithm, the best model selected based on the highest silhouette score is Hierarchical Clustering with a silhouette score of 0.552399')

        visualize_best_model(transformed_data, num_clusters_HC)

if __name__ == "__main__":
    main()
