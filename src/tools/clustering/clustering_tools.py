import numpy as np
import pandas as pd
import os
from typing import Dict, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from tools.data_loading.data_loading_utils import _load_data_from_file


def register_clustering_tools(mcp):
    """Register clustering tools for the given MCP server instance"""

    @mcp.tool()
    def kmeans_clustering(file_path: str, n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
        """
        Perform K-means clustering on data from a file.

        K-means is a centroid-based clustering algorithm that partitions data into k clusters.

        PROS:
        - Computationally efficient: O(n*k*i) where n=samples, k=clusters, i=iterations
        - Simple and well-understood algorithm
        - Works well with spherical, well-separated clusters
        - Guaranteed convergence
        - Good performance on large datasets
        - Produces compact, balanced clusters

        CONS:
        - Requires pre-specifying number of clusters (k)
        - Sensitive to the random initialization of cluster centers
        - Assumes spherical clusters of similar size
        - Sensitive to outliers (outliers can skew centroids)
        - Struggles with non-convex cluster shapes
        - Poor performance on clusters with different densities

        BEST USE CASES:
        - When you know the approximate number of clusters
        - Data with spherical, well-separated clusters
        - Large datasets where efficiency is important
        - When clusters are of similar size and density
        - As a preprocessing step for other algorithms using a k number larger than the actual

        AVOID WHEN:
        - Clusters have very different sizes or densities
        - Data contains many outliers
        - Clusters have non-spherical shapes
        - Number of clusters is completely unknown

        Args:
            file_path: Path to CSV file containing the dataset
            n_clusters: Number of clusters to find
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with cluster labels, centroids, and algorithm metadata
        """
        try:
            data_array = _load_data_from_file(file_path)
            
            # Standardize the data for better clustering
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            
            # Transform centroids back to original scale
            centroids_scaled = kmeans.cluster_centers_
            centroids = scaler.inverse_transform(centroids_scaled)
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # Calculate cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "centroid": centroids[label].tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Save results to CSV file to prevent context overflow
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_kmeans_results.csv"
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'K-means'
            })
            
            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Save metadata separately
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: K-means\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Inertia: {inertia:.4f}\n")
                f.write(f"Algorithm complexity: O(n*k*i)\n")
                f.write(f"Efficiency: High\n")
                f.write(f"Scalability: Excellent\n")
                f.write(f"Outlier sensitivity: High\n")
                f.write(f"Cluster sizes: {[cluster_stats[i]['size'] for i in range(n_clusters)]}\n")
                for i in range(n_clusters):
                    f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, centroid={cluster_stats[i]['centroid']}\n")
            
            return {
                "success": True,
                "algorithm": "K-means",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": n_clusters,
                "inertia": float(inertia),
                "file_size_bytes": os.path.getsize(results_file),
                "message": f"K-means clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": n_clusters,
                    "cluster_sizes": [cluster_stats[i]["size"] for i in range(n_clusters)],
                    "efficiency": "High"
                }
            }
        except Exception as e:
            return {"error": f"K-means clustering failed: {str(e)}"}

    @mcp.tool()
    def dbscan(file_path: str, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """
            
            
           Perform DBSCAN clustering on data from a file.

            DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are close to each other 
            based on a distance measurement (usually Euclidean) and a minimum number of points.

            PROS:
            - Does NOT require specifying the number of clusters (k) beforehand like k-mean
            - Can find arbitrarily shaped clusters (e.g., non-linear, concave, nested rings)
            - Robust to outliers and noise (explicitly labels them as -1)
            - Only requires two parameters (eps and min_samples)
            - Not affected by the order of points in the dataset

            CONS:
            - Struggles with clusters of varying densities (needs different 'eps' values)
            - Sensitive to the choice of 'eps' and 'min_samples' parameters
            - Can be computationally expensive on very large datasets without spatial indexing 
            - Struggles with high-dimensional data (Curse of Dimensionality) unless reduced
            

            BEST USE CASES:
            - Anomaly detection (finding outliers)
            - When the number of clusters is unknown
            - Data with non-spherical or complex cluster shapes
            - Noisy datasets where excluding outliers is important

            AVOID WHEN:
            - Clusters have significant differences in density
            - Data is very high-dimensional 
            - You need a specific number of clusters forced onto the data
            - computational efficiency is important

            Args:
                file_path: Path to CSV file containing the dataset
                eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

            Returns:
                Dictionary with cluster labels, statistics, and algorithm metadata
        
        
        
        """
        try:
            data_array = _load_data_from_file(file_path)
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_scaled)
            
            # Identify outliers (label = -1)
            outlier_mask = labels == -1
            outliers = data_array[outlier_mask]
            
            # Get unique cluster labels (excluding -1 for noise)
            unique_labels = np.unique(labels[labels != -1])
            n_clusters = len(unique_labels)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "center": cluster_points.mean(axis=0).tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Save results to CSV file to prevent context overflow
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_dbscan_results.csv"
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'DBSCAN'
            })
            
            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Save metadata separately
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: DBSCAN\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Number of outliers: {np.sum(outlier_mask)}\n")
                f.write(f"Cluster label for outliers: {-1}\n")
                f.write(f"Outliers: {outliers}")
                f.write(f"Parameters: eps={eps}, min_samples={min_samples}\n")
                f.write(f"Algorithm complexity: O(n log n)\n")
                f.write(f"Efficiency: Medium\n")
                f.write(f"Scalability: Good\n")
                f.write(f"Outlier sensitivity: Low (robust)\n")
                if n_clusters > 0:
                    cluster_sizes = [cluster_stats[i]['size'] for i in range(n_clusters)]
                    f.write(f"Cluster sizes: {cluster_sizes}\n")
                    for i in range(n_clusters):
                        f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, center={cluster_stats[i]['center']}\n")
            
            return {
                "success": True,
                "algorithm": "DBSCAN",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": int(n_clusters),
                "n_outliers": int(np.sum(outlier_mask)),
                "file_size_bytes": os.path.getsize(results_file),
                "parameters": {"eps": eps, "min_samples": min_samples},
                "message": f"DBSCAN clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": int(n_clusters),
                    "outliers_detected": int(np.sum(outlier_mask)),
                    "efficiency": "Medium"
                }
            }
        except Exception as e:
            return {"error": f"DBSCAN clustering failed: {str(e)}"}

    @mcp.tool()
    def hierarchical_single_linkage(file_path: str, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform Hierarchical Clustering using Single Linkage (MIN) method.

        Single Linkage is an agglomerative hierarchical clustering method that merges clusters
        based on the minimum distance between points in different clusters. Produces elongated clusters.

        PROS:
        - Can discover clusters of arbitrary shapes (non-convex, elongated)
        - Produces hierarchical structure (dendrogram) for analysis
        - Deterministic (no random initialization)
        - Does not assume spherical clusters
        - Works well with irregular, chain-like cluster structures

        CONS:
        - Highly sensitive to noise and outliers (chaining problem)
        - Tends to create long, thin clusters even when not appropriate
        - Computationally expensive: O(n³) time, O(n²) space
        - Cannot undo merges (greedy algorithm)
        - Struggles with balanced, compact clusters

        BEST USE CASES:
        - Discovering non-spherical, elongated cluster structures
        - When you need hierarchical relationships between clusters
        - Datasets with chain-like or filamentary structures
        - Exploratory data analysis to understand data hierarchy

        AVOID WHEN:
        - Data contains significant noise or outliers
        - Clusters are expected to be compact and spherical
        - Working with very large datasets (computationally expensive)
        - Need robust results not affected by noise

        Args:
            file_path: Path to CSV file containing the dataset
            n_clusters: Number of clusters to find

        Returns:
            Dictionary with cluster labels, statistics, and algorithm metadata
        """
        try:
            data_array = _load_data_from_file(file_path)

            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)

            # Perform Hierarchical Clustering with Single Linkage
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='single',
                metric='euclidean'
            )
            labels = hierarchical.fit_predict(data_scaled)

            # Calculate cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "center": cluster_points.mean(axis=0).tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }

            # Create reports directory
            os.makedirs('reports', exist_ok=True)

            # Save results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_hierarchical_single_results.csv"

            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'Hierarchical-Single-Linkage'
            })

            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Save metadata
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: Hierarchical Clustering (Single Linkage / MIN)\n")
                f.write(f"Type: Agglomerative (Bottom-up)\n")
                f.write(f"Linkage Method: Single (Minimum distance between clusters)\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Algorithm complexity: O(n³) time, O(n²) space\n")
                f.write(f"Efficiency: Low (for large datasets)\n")
                f.write(f"Scalability: Poor\n")
                f.write(f"Outlier sensitivity: Very High (chaining effect)\n")
                f.write(f"Cluster shape: Elongated, chain-like\n")
                f.write(f"Produces hierarchy: Yes (dendrogram)\n")
                f.write(f"Cluster sizes: {[cluster_stats[i]['size'] for i in range(n_clusters)]}\n")
                for i in range(n_clusters):
                    f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, center={cluster_stats[i]['center']}\n")

            return {
                "success": True,
                "algorithm": "Hierarchical-Single-Linkage",
                "linkage_type": "single (MIN)",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": n_clusters,
                "file_size_bytes": os.path.getsize(results_file),
                "message": f"Hierarchical Single Linkage clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": n_clusters,
                    "cluster_sizes": [cluster_stats[i]["size"] for i in range(n_clusters)],
                    "efficiency": "Low",
                    "cluster_shape": "Elongated, chain-like"
                }
            }
        except Exception as e:
            return {"error": f"Hierarchical Single Linkage clustering failed: {str(e)}"}

    @mcp.tool()
    def hierarchical_complete_linkage(file_path: str, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform Hierarchical Clustering using Complete Linkage (MAX) method.

        Complete Linkage is an agglomerative hierarchical clustering method that merges clusters
        based on the maximum distance between points in different clusters. Produces compact, spherical clusters.

        PROS:
        - Robust to noise and outliers (opposite of single linkage)
        - Produces compact, well-separated clusters
        - Deterministic (no random initialization)
        - Produces hierarchical structure (dendrogram)
        - Less affected by chaining problem
        - More balanced cluster sizes than single linkage

        CONS:
        - Cannot discover non-spherical, elongated clusters
        - Biased toward finding globular clusters of similar size
        - Computationally expensive: O(n³) time, O(n²) space
        - Cannot undo merges (greedy algorithm)
        - May break natural elongated clusters into pieces

        BEST USE CASES:
        - When clusters are expected to be compact and spherical
        - Data contains noise or outliers (more robust than single linkage)
        - Need hierarchical relationships between clusters
        - Exploratory data analysis with dendrogram visualization
        - Medium-sized datasets (< 10,000 points)

        AVOID WHEN:
        - Clusters have elongated or irregular shapes
        - Working with very large datasets (computationally expensive)
        - Clusters have very different sizes or densities
        - Need real-time clustering

        Args:
            file_path: Path to CSV file containing the dataset
            n_clusters: Number of clusters to find

        Returns:
            Dictionary with cluster labels, statistics, and algorithm metadata
        """
        try:
            data_array = _load_data_from_file(file_path)

            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)

            # Perform Hierarchical Clustering with Complete Linkage
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='complete',
                metric='euclidean'
            )
            labels = hierarchical.fit_predict(data_scaled)

            # Calculate cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "center": cluster_points.mean(axis=0).tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }

            # Create reports directory
            os.makedirs('reports', exist_ok=True)

            # Save results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_hierarchical_complete_results.csv"

            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'Hierarchical-Complete-Linkage'
            })

            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Save metadata
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: Hierarchical Clustering (Complete Linkage / MAX)\n")
                f.write(f"Type: Agglomerative (Bottom-up)\n")
                f.write(f"Linkage Method: Complete (Maximum distance between clusters)\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Algorithm complexity: O(n³) time, O(n²) space\n")
                f.write(f"Efficiency: Low (for large datasets)\n")
                f.write(f"Scalability: Poor\n")
                f.write(f"Outlier sensitivity: Low (robust)\n")
                f.write(f"Cluster shape: Compact, spherical\n")
                f.write(f"Produces hierarchy: Yes (dendrogram)\n")
                f.write(f"Cluster sizes: {[cluster_stats[i]['size'] for i in range(n_clusters)]}\n")
                for i in range(n_clusters):
                    f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, center={cluster_stats[i]['center']}\n")

            return {
                "success": True,
                "algorithm": "Hierarchical-Complete-Linkage",
                "linkage_type": "complete (MAX)",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": n_clusters,
                "file_size_bytes": os.path.getsize(results_file),
                "message": f"Hierarchical Complete Linkage clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": n_clusters,
                    "cluster_sizes": [cluster_stats[i]["size"] for i in range(n_clusters)],
                    "efficiency": "Low",
                    "cluster_shape": "Compact, spherical"
                }
            }
        except Exception as e:
            return {"error": f"Hierarchical Complete Linkage clustering failed: {str(e)}"}

    @mcp.tool()
    def hierarchical_average_linkage(file_path: str, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform Hierarchical Clustering using Average Linkage (UPGMA) method.

        Average Linkage is an agglomerative hierarchical clustering method that merges clusters
        based on the average distance between all pairs of points. Balances between single and complete linkage.

        PROS:
        - Good balance between single and complete linkage
        - More robust to outliers than single linkage
        - Less biased toward spherical clusters than complete linkage
        - Produces hierarchical structure (dendrogram)
        - Deterministic (no random initialization)
        - Generally produces reasonable cluster shapes

        CONS:
        - Computationally expensive: O(n³) time, O(n²) space
        - Cannot undo merges (greedy algorithm)
        - Requires specifying number of clusters in advance
        - Not as robust to outliers as complete linkage
        - Not as good at finding elongated clusters as single linkage

        BEST USE CASES:
        - When cluster shape is unknown or mixed
        - General-purpose hierarchical clustering (balanced approach)
        - Data with moderate noise
        - Need hierarchical relationships between clusters
        - Medium-sized datasets (< 10,000 points)

        AVOID WHEN:
        - Working with very large datasets (computationally expensive)
        - Clusters are known to be extremely elongated (use single)
        - Need maximum robustness to outliers (use complete)
        - Real-time clustering required

        Args:
            file_path: Path to CSV file containing the dataset
            n_clusters: Number of clusters to find

        Returns:
            Dictionary with cluster labels, statistics, and algorithm metadata
        """
        try:
            data_array = _load_data_from_file(file_path)

            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)

            # Perform Hierarchical Clustering with Average Linkage
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average',
                metric='euclidean'
            )
            labels = hierarchical.fit_predict(data_scaled)

            # Calculate cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "center": cluster_points.mean(axis=0).tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }

            # Create reports directory
            os.makedirs('reports', exist_ok=True)

            # Save results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_hierarchical_average_results.csv"

            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'Hierarchical-Average-Linkage'
            })

            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Save metadata
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: Hierarchical Clustering (Average Linkage / UPGMA)\n")
                f.write(f"Type: Agglomerative (Bottom-up)\n")
                f.write(f"Linkage Method: Average (Mean of all pairwise distances)\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Algorithm complexity: O(n³) time, O(n²) space\n")
                f.write(f"Efficiency: Low (for large datasets)\n")
                f.write(f"Scalability: Poor\n")
                f.write(f"Outlier sensitivity: Medium (balanced)\n")
                f.write(f"Cluster shape: Moderately compact, flexible\n")
                f.write(f"Produces hierarchy: Yes (dendrogram)\n")
                f.write(f"Cluster sizes: {[cluster_stats[i]['size'] for i in range(n_clusters)]}\n")
                for i in range(n_clusters):
                    f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, center={cluster_stats[i]['center']}\n")

            return {
                "success": True,
                "algorithm": "Hierarchical-Average-Linkage",
                "linkage_type": "average (UPGMA)",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": n_clusters,
                "file_size_bytes": os.path.getsize(results_file),
                "message": f"Hierarchical Average Linkage clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": n_clusters,
                    "cluster_sizes": [cluster_stats[i]["size"] for i in range(n_clusters)],
                    "efficiency": "Low",
                    "cluster_shape": "Moderately compact, balanced"
                }
            }
        except Exception as e:
            return {"error": f"Hierarchical Average Linkage clustering failed: {str(e)}"}

    @mcp.tool()
    def hierarchical_ward(file_path: str, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform Hierarchical Clustering using Ward's method (minimum variance).

        Ward's method is an agglomerative hierarchical clustering algorithm that minimizes
        within-cluster variance. Produces balanced, spherical clusters similar to K-means.

        PROS:
        - Produces well-balanced, compact clusters
        - Minimizes within-cluster variance (optimal criterion)
        - Generally produces clusters of similar size
        - Works very well for spherical, well-separated data
        - Produces hierarchical structure (dendrogram)
        - Deterministic (no random initialization)
        - Most popular hierarchical method in practice

        CONS:
        - Strongly biased toward spherical clusters of similar size
        - Cannot handle elongated or irregularly shaped clusters
        - Sensitive to outliers (minimizing variance affected by extremes)
        - Computationally expensive: O(n³) time, O(n²) space
        - Cannot undo merges (greedy algorithm)
        - Only works with Euclidean distance

        BEST USE CASES:
        - When clusters are expected to be spherical and balanced
        - Need hierarchical relationships AND good cluster quality
        - Want similar results to K-means but with dendrogram
        - Exploratory data analysis with moderate-sized datasets
        - Medium-sized datasets (< 10,000 points)

        AVOID WHEN:
        - Clusters have very different sizes
        - Clusters have elongated or irregular shapes
        - Data contains many outliers
        - Working with very large datasets (computationally expensive)
        - Need to use non-Euclidean distance metrics

        Args:
            file_path: Path to CSV file containing the dataset
            n_clusters: Number of clusters to find

        Returns:
            Dictionary with cluster labels, statistics, and algorithm metadata
        """
        try:
            data_array = _load_data_from_file(file_path)

            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)

            # Perform Hierarchical Clustering with Ward's method
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                metric='euclidean'  # Ward only works with Euclidean distance
            )
            labels = hierarchical.fit_predict(data_scaled)

            # Calculate cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "center": cluster_points.mean(axis=0).tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }

            # Create reports directory
            os.makedirs('reports', exist_ok=True)

            # Save results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_hierarchical_ward_results.csv"

            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'Hierarchical-Ward'
            })

            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Save metadata
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: Hierarchical Clustering (Ward's method)\n")
                f.write(f"Type: Agglomerative (Bottom-up)\n")
                f.write(f"Linkage Method: Ward (Minimum Variance)\n")
                f.write(f"Optimization criterion: Minimize within-cluster sum of squares\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Algorithm complexity: O(n³) time, O(n²) space\n")
                f.write(f"Efficiency: Low (for large datasets)\n")
                f.write(f"Scalability: Poor\n")
                f.write(f"Outlier sensitivity: Medium-High\n")
                f.write(f"Cluster shape: Spherical, balanced\n")
                f.write(f"Produces hierarchy: Yes (dendrogram)\n")
                f.write(f"Distance metric: Euclidean (required)\n")
                f.write(f"Cluster sizes: {[cluster_stats[i]['size'] for i in range(n_clusters)]}\n")
                for i in range(n_clusters):
                    f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, center={cluster_stats[i]['center']}\n")

            return {
                "success": True,
                "algorithm": "Hierarchical-Ward",
                "linkage_type": "ward (minimum variance)",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": n_clusters,
                "file_size_bytes": os.path.getsize(results_file),
                "message": f"Hierarchical Ward clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": n_clusters,
                    "cluster_sizes": [cluster_stats[i]["size"] for i in range(n_clusters)],
                    "efficiency": "Low",
                    "cluster_shape": "Spherical, balanced"
                }
            }
        except Exception as e:
            return {"error": f"Hierarchical Ward clustering failed: {str(e)}"}
