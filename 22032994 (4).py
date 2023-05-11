import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

import wbdata

# Set the default action for all future warnings to "ignore"
warnings.simplefilter("ignore", FutureWarning)

def get_world_bank_data(indicators):
    """
    Retrieves World Bank data for the specified indicators and countries.

    Args:
        indicators (dict): A dictionary mapping indicator names to their codes.

    Returns:
        df (pandas.DataFrame): The original data with indicators as columns and years as index.
        df_scaled (pandas.DataFrame): The scaled data with indicators as columns and years as index.
    """
    # Retrieve the data for the indicators and countries
    data = wbdata.get_dataframe(indicators, country="all").reset_index(level=1)
    data['date'] = data['date'].astype('int')
    
    # Filter the data for the desired date range
    data = data[data.date.isin(range(1990, 2019))]
    
    # Pivot the data to have indicators as columns and years as index
    df = data.pivot(columns='date', values=list(indicators.values()))
    
    # Fill missing values using forward fill
    df = df.fillna(method='ffill', axis=1)
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = np.nan_to_num(scaler.fit_transform(df))
    df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    
    return df, df_scaled

def plot_elbow_curve(data, title=''):
    """
    Plots the elbow curve for K-means clustering.

    Args:
        data (numpy.ndarray): The data to be clustered.
        title (str, optional): The title of the plot. Defaults to an empty string.
    """
    wcss = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i, init="k-means++", random_state=2023)
        model.fit(data)
        wcss.append(model.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title(title)
    plt.savefig(f"elbow-curve-{''.join(title.split())}.png", dpi=300, transparent=True)
    plt.close()

def reduce_dim(data, ncomp):
    """
    Reduces the dimensionality of the input data using t-SNE.

    Args:
        data (numpy.ndarray): The data to be dimensionally reduced.
        ncomp (int): The number of components for t-SNE.

    Returns:
        numpy.ndarray: The reduced-dimensional data.
    """
    # Run t-SNE on the data and reduce the dimensions
    tsne = TSNE(n_components=2)
    scaler = MinMaxScaler()
    reduced_data = scaler.fit_transform(tsne.fit_transform(data))
    return reduced_data

def cluster_data(data, n_clusters):
    """
    Performs K-means clustering on the input data.

    Args:
        data (numpy.ndarray): The data to be clustered.
        n_clusters (int): The number of clusters for K-means.

    Returns:
        labels (numpy.ndarray): The cluster labels for each data point.
        centers (numpy.ndarray): The centroid coordinates of the clusters.
    """
    # Perform K-means clustering with k clusters on the given data
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=2023)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return labels, centers

def plot_clusters(data, labels, centers, title):
    """
    Plots the clusters and their centers.

    Args:
        data (numpy.ndarray): The data points.
        labels (numpy.ndarray): The cluster labels for each data point.
        centers (numpy.ndarray): The centroid coordinates of the clusters.
        title (str): The title of the plot.
    """
    x, y = data[:, 0], data[:, 1]

    # Create a scatter plot for each cluster
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        plt.scatter(x[indices], y[indices], label=f'Cluster {label}')

    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.legend()
    plt.title(f'KMeans Clustering of {title} with 2 dimensions')
    title_str = ''.join(title.split())
    plt.savefig(f'kmeans-{title_str}.png', dpi=300, transparent=True)
    plt.close()

def exponential_growth(x, a, b, c):
    return a * x + b * x**2 + c

def plot_growth_charts(cluster, country, label):
    """
    Plots the growth chart for a specific country and indicator.

    Args:
        cluster (pandas.DataFrame): The data cluster.
        country (str): The name of the country.
        label (str): The label of the indicator.
    """
    # Fit the model to the data
    xdata = cluster.loc[country][:-1].index.get_level_values(1)
    ydata = cluster.loc[country][:-1].values

    # Handle NaN values
    if np.isnan(ydata).any():
        ydata = np.nan_to_num(ydata, nan=-1, posinf=-1, neginf=-1)
    
    # Perform curve fitting
    popt, pcov = curve_fit(exponential_growth, xdata, ydata)

    # Generate predictions for future years
    xnew = np.arange(min(xdata), max(xdata)+5, 1)
    ynew = exponential_growth(xnew, *popt)

    # Plot the best fitting function and the data points
    plt.plot(xnew, ynew, 'r-', label='Best fitting function')
    plt.scatter(xdata, ydata, label='Data')
    plt.xlabel('Year')
    plt.ylabel(label)
    plt.legend()
    label_str = ''.join(label.split())
    plt.title(f'{label} - {country}')
    plt.savefig(f'{country}-{label_str}.png', dpi=300, transparent=True)
    plt.close()

def get_cluster_bounds(cluster_labels, cluster_centers, data):
    """
    Retrieves the indices of the closest and farthest data points to each cluster center.

    Args:
        cluster_labels (numpy.ndarray): The cluster labels for each data point.
        cluster_centers (numpy.ndarray): The centroid coordinates of the clusters.
        data (numpy.ndarray): The data points.

    Returns:
        closest_indices (numpy.ndarray): The indices of the closest data points to each cluster center.
        farthest_indices (numpy.ndarray): The indices of the farthest data points to each cluster center.
    """
    closest_indices = np.zeros(len(np.unique(cluster_labels)), dtype=int)
    farthest_indices = np.zeros(len(np.unique(cluster_labels)), dtype=int)

    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        distances = np.linalg.norm(data[indices] - cluster_centers[label], axis=1)
        closest_indices[label] = indices[np.argmin(distances)]
        farthest_indices[label] = indices[np.argmax(distances)]

    return closest_indices, farthest_indices

# Define the indicators
indicators = {'EN.ATM.CO2E.PC': 'CO2 emissions per capita', 'AG.LND.FRST.ZS': 'Forest area as % of land area'}

# Retrieve the data for CO2 emissions per capita
df_emissions, df_emissions_scaled = get_world_bank_data({'EN.ATM.CO2E.PC': 'CO2 emissions per capita'})

# Plot the elbow curve for CO2 emissions per capita
plot_elbow_curve(df_emissions_scaled, title="CO2 emissions per capita")

# Reduce the dimensionality of the data
emissions_reduced = reduce_dim(df_emissions_scaled, ncomp=2)

# Perform clustering on the reduced data
cluster_labels_, cluster_centers_ = cluster_data(emissions_reduced, n_clusters=3)

# Plot the clusters
plot_clusters(emissions_reduced, cluster_labels_, cluster_centers_, title='CO2 emissions per capita')

# Assign cluster labels to the original data
df_emissions['Cluster'] = cluster_labels_

# Separate the data by clusters
cluster_0 = df_emissions[df_emissions['Cluster'] == 0]
cluster_1 = df_emissions[df_emissions['Cluster'] == 1]
cluster_2 = df_emissions[df_emissions['Cluster'] == 2]


# Retrieve the indices of the closest and farthest data points in cluster 1
closest_indices, farthest_indices = get_cluster_bounds(cluster_labels_, cluster_centers_, emissions_reduced)

# Get the country names
closest_country = df_emissions.iloc[closest_indices[1]].name
farthest_country = df_emissions.iloc[farthest_indices[1]].name

# Plot growth charts for the closest and farthest countries in cluster 1
plot_growth_charts(cluster_1, closest_country, label="CO2 Emissions per capita")
plot_growth_charts(cluster_1, farthest_country, label="CO2 Emissions per capita")

# Retrieve the data for forest area as % of land area
df_forest_area, df_forest_area_scaled = get_world_bank_data({'AG.LND.FRST.ZS': 'Forest area as % of land area'})

# Plot growth charts for specific countries related to forest area
plot_growth_charts(df_forest_area_scaled, "Comoros", label="Forest area as % of land area")
plot_growth_charts(df_forest_area_scaled.iloc[:, 2:], "Georgia", label="Forest area as % of land area")


# Compute cluster sizes
cluster_sizes = pd.Series(cluster_labels_).value_counts().sort_index()

# Compute variance within each cluster
variance = pd.DataFrame(df_emissions).groupby(cluster_labels_).var()

# Compute mean variance for each cluster
variance_means = variance.mean(axis=1)

# Create a tabular representation of cluster statistics
cluster_stats = pd.DataFrame({
    'Cluster': range(3),
    'Size': cluster_sizes,
    'Centroid': cluster_centers_.tolist(),
    'Variance': variance_means.tolist(),
})
cluster_stats = cluster_stats.set_index('Cluster')

# Save cluster statistics to a CSV file
cluster_stats.to_csv("cluster_stats.csv", index=None)

# Calculate the sum of CO2 emissions for specific countries
co2_emissions_sum = df_emissions.loc[['United States', 'Russian Federation', 'United Kingdom', 'India', 'China', 'South Africa']].sum(axis=1)

# Save CO2 emissions sums to a CSV file
co2_emissions_sum.to_csv("co2Emissions.csv", header=None)

