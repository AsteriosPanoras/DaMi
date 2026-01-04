import numpy as np
import pandas as pd
import os
import sys
import time
from typing import Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add parent directory to path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import config

from tools.data_loading.data_loading_utils import _load_data_from_file


def register_outlier_detection_tools(mcp):
    """Register outlier detection tools for the given MCP server instance"""

    @mcp.tool()
    def detect_outliers_density_based(
        file_path: str,
        n_micro: int = None,
        n_macro: int = None,
        percentile: int = 90,
        min_radius: float = 0.5,
        max_radius: float = 5.0,
        fixed_sigma: float = None
    ) -> Dict[str, Any]:
        """
        Perform density-based outlier detection using two-stage clustering.

        This algorithm uses a sophisticated two-stage clustering approach combined with
        density-based filtering to identify outliers in 2D datasets. It's particularly
        effective at finding anomalies in data with varying cluster densities.

        ALGORITHM OVERVIEW:
        1. PREPROCESSING: Standardizes data and removes duplicates/NaN/infinities
        2. MICRO-CLUSTERING: Creates many small clusters (default 100) to capture local patterns
        3. DENSITY FILTERING: Removes sparse micro-clusters below dynamic threshold
        4. MACRO-CLUSTERING: Groups valid micro-clusters into final clusters (default 5)
        5. HYBRID OUTLIER DETECTION: Uses micro-distance, macro-distance, and adaptive radius
        6. SAFETY SHIELD: Adaptive per-cluster radius prevents false positives

        PROS:
        - Robust to varying cluster densities (adaptive radius per cluster)
        - Combines local (micro) and global (macro) distance measures
        - Dynamic threshold based on cluster statistics (not fixed)
        - Handles sparse and dense regions differently
        - Low false positive rate thanks to "safety shield" mechanism
        - Works well with non-spherical clusters
        - Automatically filters noise during micro-clustering stage

        CONS:
        - Requires two hyperparameters (n_micro, n_macro) to be tuned
        - More computationally expensive than simple methods (two K-means runs)
        - May struggle if all clusters have uniform density (overhead without benefit)
        - Fixed to 5 macro clusters as per exercise requirements
        - Assumes 2D data (designed for X,Y coordinates)

        BEST USE CASES:
        - Datasets with clusters of varying densities
        - When you need both local and global anomaly detection
        - Contaminated datasets with noise and true outliers
        - Spatial data with geographic coordinates
        - When false positives are costly (e.g., flagging good customers as fraudulent)
        - Exercise requirement: detecting outliers in corrupted 2D point datasets

        AVOID WHEN:
        - Clusters have uniform density (simpler methods like DBSCAN may suffice)
        - Very high-dimensional data (curse of dimensionality)
        - Real-time applications requiring instant responses
        - Very small datasets (< 100 points) where micro-clustering is overkill
        - When interpretability is more important than accuracy

        PARAMETERS EXPLAINED:
        - n_micro: Number of micro-clusters (more = finer granularity, slower)
        - n_macro: Number of final clusters (exercise requires 5)
        - percentile: Base radius calculation (90 = include 90% of cluster points)
        - min_radius/max_radius: Safety bounds for adaptive radius
        - fixed_sigma: Threshold multiplier for micro-distance (higher = fewer outliers)

        DETECTION STRATEGY:
        A point is flagged as outlier if:
        1. It's far from its micro-cluster center (> mean + sigma*std), OR
        2. It belongs to a sparse micro-cluster (size < dynamic threshold)
        AND
        3. It's beyond the adaptive safety radius of its macro-cluster

        Args:
            file_path: Path to CSV file containing the 2D dataset
            n_micro: Number of micro-clusters for stage 1 (default: 200)
            n_macro: Number of final macro-clusters (default: 5, per exercise)
            percentile: Percentile for adaptive radius calculation (default: 90)
            min_radius: Minimum safety radius for any cluster (default: 0.5)
            max_radius: Maximum safety radius for any cluster (default: 5.0)
            fixed_sigma: Sigma multiplier for outlier threshold (default: 3.5)

        Returns:
            Dictionary with outlier detection results, statistics, and file paths
        """
        try:
            # Use config defaults if not specified
            if n_micro is None:
                n_micro = config.DEFAULT_N_MICRO
            if n_macro is None:
                n_macro = config.DEFAULT_N_MACRO
            if fixed_sigma is None:
                fixed_sigma = config.FIXED_SIGMA

            # Load data
            data_array = _load_data_from_file(file_path)
            df_raw = pd.DataFrame(data_array, columns=['x', 'y'])

            start_time = time.time()

            # ====================================================================
            # PHASE 1: PREPROCESSING
            # ====================================================================
            initial_rows = len(df_raw)
            df_clean = df_raw.copy()

            # Convert to numeric
            df_clean['x'] = pd.to_numeric(df_clean['x'], errors='coerce')
            df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')

            # Handle infinite values (with same logic as standalone script)
            try:
                temp_vals = df_clean[['x', 'y']].astype(float)
                num_inf = np.isinf(temp_vals).values.sum()
                if num_inf > 0:
                    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            except:
                df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop NaN
            df_clean.dropna(inplace=True)

            # Smart deduplication
            temp_rounded = df_clean.round(6)
            duplicates_mask = temp_rounded.duplicated()
            num_dupes = duplicates_mask.sum()
            if num_dupes > 0:
                df_clean = df_clean[~duplicates_mask]

            # Save original indices BEFORE reset_index
            df_clean['orig_index'] = df_clean.index
            df_clean.reset_index(drop=True, inplace=True)

            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean[['x', 'y']].values)

            # ====================================================================
            # PHASE 2: TWO-STAGE CLUSTERING
            # ====================================================================

            # Stage 1: Micro-Clustering
            kmeans_micro = KMeans(
                n_clusters=n_micro,
                random_state=config.KMEANS_RANDOM_STATE,
                n_init=config.KMEANS_N_INIT_MICRO
            )
            micro_labels = kmeans_micro.fit_predict(X_scaled)
            micro_centers = kmeans_micro.cluster_centers_

            # Calculate dynamic minimum micro-cluster size
            unique_micro, counts_micro = np.unique(micro_labels, return_counts=True)
            median_size = np.median(counts_micro)
            MIN_MICRO_SIZE = max(config.MIN_MICRO_SIZE, int(median_size * 0.2))  # 20% of median

            # Density Filtering
            micro_counts_map = dict(zip(unique_micro, counts_micro))
            valid_micro_indices = []
            valid_weights = []

            for i in range(n_micro):
                if micro_counts_map.get(i, 0) >= MIN_MICRO_SIZE:
                    valid_micro_indices.append(i)
                    valid_weights.append(micro_counts_map.get(i, 0))

            if len(valid_micro_indices) < n_macro:
                valid_micro_centers = micro_centers
            else:
                valid_micro_centers = micro_centers[valid_micro_indices]

            # Stage 2: Macro-Clustering
            kmeans_macro = KMeans(
                n_clusters=n_macro,
                random_state=config.KMEANS_RANDOM_STATE,
                n_init=config.KMEANS_N_INIT_MACRO
            )
            kmeans_macro.fit(valid_micro_centers, sample_weight=valid_weights)
            final_centers = kmeans_macro.cluster_centers_

            # Assign final labels
            macro_labels_for_all_micro = kmeans_macro.predict(micro_centers)
            final_labels = macro_labels_for_all_micro[micro_labels]

            # ====================================================================
            # PHASE 3: HYBRID OUTLIER DETECTION
            # ====================================================================

            # Calculate distances
            my_micro_centers = micro_centers[micro_labels]
            micro_distances = np.linalg.norm(X_scaled - my_micro_centers, axis=1)

            my_macro_centers = final_centers[final_labels]
            macro_distances = np.linalg.norm(X_scaled - my_macro_centers, axis=1)

            # Density rule
            is_isolated_mask = np.array([micro_counts_map[label] < MIN_MICRO_SIZE for label in micro_labels])

            # Create analysis DataFrame
            analysis_df = pd.DataFrame({
                'cluster': final_labels,
                'micro_dist': micro_distances,
                'macro_dist': macro_distances,
                'is_isolated': is_isolated_mask,
                'scaled_x': X_scaled[:, 0],
                'scaled_y': X_scaled[:, 1]
            })

            # Calculate adaptive safety radius (density-based)
            safety_radii_dict = {}
            for cluster_id in np.unique(final_labels):
                cluster_mask = (final_labels == cluster_id)
                cluster_points = X_scaled[cluster_mask]
                center = final_centers[cluster_id]

                distances = np.linalg.norm(cluster_points - center, axis=1)
                radius = np.percentile(distances, percentile)

                # Density factor
                mean_dist = np.mean(distances)
                density_factor = 1.0 / (1.0 + 0.5 * mean_dist)
                radius *= density_factor

                # Size penalty
                if len(cluster_points) < 50:
                    radius *= 0.80

                # Clip to bounds
                radius = np.clip(radius, min_radius, max_radius)
                safety_radii_dict[cluster_id] = radius

            analysis_df['safety_radius'] = analysis_df['cluster'].map(safety_radii_dict)

            # Calculate thresholds
            cluster_stats = analysis_df.groupby('cluster')['micro_dist'].agg(['mean', 'std']).reset_index()
            cluster_stats['std'] = cluster_stats['std'].fillna(0)
            analysis_df = analysis_df.merge(cluster_stats, on='cluster', how='left')
            analysis_df['threshold'] = analysis_df['mean'] + (fixed_sigma * analysis_df['std'])

            # Final decision
            suspects = (analysis_df['micro_dist'] > analysis_df['threshold']) | (analysis_df['is_isolated'] == True)
            analysis_df['is_outlier'] = (suspects &
                                        (analysis_df['macro_dist'] > analysis_df['safety_radius'])).astype(int)

            end_time = time.time()

            # ====================================================================
            # PHASE 4: RESULTS & VISUALIZATION
            # ====================================================================

            num_outliers = analysis_df['is_outlier'].sum()
            n_saved = np.sum(suspects & (analysis_df['macro_dist'] <= analysis_df['safety_radius']))

            # Get base filename and create organized output directory
            base_filename = os.path.basename(file_path)
            if base_filename in ['data.txt', 'data.csv', 'ground_truth.csv']:
                parent_folder = os.path.basename(os.path.dirname(file_path))
                dataset_name = parent_folder
            else:
                dataset_name = os.path.splitext(base_filename)[0]

            # Use config to get organized output paths
            output_dir = config.get_output_path('detection_noPCA_density', dataset_name)

            # Save outliers to CSV
            outlier_indices = analysis_df.index[analysis_df['is_outlier'] == 1]
            orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values
            outliers_df = df_raw.iloc[orig_indices]

            outliers_file = os.path.join(output_dir, f"{dataset_name}_outliers.csv")
            outliers_df.to_csv(outliers_file, index=False, header=False, encoding='utf-8')

            # Save full results
            results_file = os.path.join(output_dir, f"{dataset_name}_results.csv")
            results_df = pd.DataFrame({
                'point_id': range(len(analysis_df)),
                'x': df_clean['x'].values,
                'y': df_clean['y'].values,
                'cluster_label': final_labels,
                'is_outlier': analysis_df['is_outlier'].values,
                'micro_distance': micro_distances,
                'macro_distance': macro_distances,
                'source_file': file_path,
                'algorithm': 'Density-Based Two-Stage'
            })
            results_df.to_csv(results_file, index=False, encoding='utf-8')

            # Generate visualizations
            _generate_outlier_plots(
                analysis_df, final_centers, df_raw, df_clean,
                dataset_name, output_dir
            )

            # Create metadata
            metadata_file = os.path.join(output_dir, f"{dataset_name}_metadata.txt")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Algorithm: Density-Based Two-Stage Outlier Detection\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Processing time: {end_time - start_time:.4f} seconds\n")
                f.write(f"Initial rows: {initial_rows}\n")
                f.write(f"After cleaning: {len(df_clean)}\n")
                f.write(f"Duplicates removed: {num_dupes}\n")
                f.write(f"Micro-clusters: {n_micro}\n")
                f.write(f"Macro-clusters: {n_macro}\n")
                f.write(f"Dynamic min micro size: {MIN_MICRO_SIZE}\n")
                f.write(f"Fixed sigma: {fixed_sigma}\n")
                f.write(f"Outliers detected: {num_outliers} ({num_outliers/len(df_clean)*100:.2f}%)\n")
                f.write(f"False positives prevented: {n_saved}\n")
                f.write(f"\nCluster Statistics:\n")
                f.write(f"{'ID':<5} {'Count':<8} {'SafetyRadius':<15} {'Outliers':<10}\n")
                f.write("-" * 50 + "\n")
                for i in range(n_macro):
                    stats = analysis_df[analysis_df['cluster'] == i]
                    if stats.empty:
                        continue
                    n_out = stats['is_outlier'].sum()
                    radius = safety_radii_dict[i]
                    f.write(f"{i:<5} {len(stats):<8} {radius:<15.4f} {n_out:<10}\n")

            return {
                "success": True,
                "algorithm": "Density-Based Two-Stage",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "outliers_file": os.path.abspath(outliers_file),
                "metadata_file": os.path.abspath(metadata_file),
                "processing_time_seconds": float(end_time - start_time),
                "parameters": {
                    "n_micro": n_micro,
                    "n_macro": n_macro,
                    "percentile": percentile,
                    "min_radius": min_radius,
                    "max_radius": max_radius,
                    "fixed_sigma": fixed_sigma
                },
                "statistics": {
                    "initial_rows": initial_rows,
                    "cleaned_rows": len(df_clean),
                    "duplicates_removed": int(num_dupes),
                    "outliers_detected": int(num_outliers),
                    "outlier_percentage": float(num_outliers/len(df_clean)*100),
                    "false_positives_prevented": int(n_saved),
                    "dynamic_min_micro_size": MIN_MICRO_SIZE
                },
                "message": f"Outlier detection completed. Found {num_outliers} outliers ({num_outliers/len(df_clean)*100:.2f}%). Results saved to {results_file}",
                "plots": {
                    "scaled_space": os.path.join(output_dir, f"{dataset_name}_plot_scaled.png"),
                    "original_space": os.path.join(output_dir, f"{dataset_name}_plot_raw.png")
                }
            }

        except Exception as e:
            return {"error": f"Outlier detection failed: {str(e)}"}


def _generate_outlier_plots(analysis_df, centers, df_raw, df_clean, dataset_name, output_dir):
    """Generate visualization plots for outlier detection"""

    # Plot 1: Scaled space (algorithm workspace)
    plt.figure(figsize=(12, 8))

    inliers = analysis_df[analysis_df['is_outlier'] == 0]
    outliers = analysis_df[analysis_df['is_outlier'] == 1]

    plt.scatter(inliers['scaled_x'], inliers['scaled_y'],
                c=inliers['cluster'], cmap='viridis', s=20, alpha=0.6, label='Inliers')

    if not outliers.empty:
        plt.scatter(outliers['scaled_x'], outliers['scaled_y'],
                    c='red', marker='x', s=80, linewidth=2, label='Outliers')

    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=300, marker='*', edgecolors='white', label='Centroids')

    plt.title(f"Two-Stage Clustering (Density-Based)\nDataset: {dataset_name}")
    plt.xlabel("Scaled X")
    plt.ylabel("Scaled Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    scaled_plot = os.path.join(output_dir, f"{dataset_name}_plot_scaled.png")
    plt.savefig(scaled_plot, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Original space (raw data)
    plt.figure(figsize=(12, 8))

    outlier_indices = analysis_df.index[analysis_df['is_outlier'] == 1]
    orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

    all_indices = df_raw.index
    inlier_mask = ~all_indices.isin(orig_indices)

    inliers_x = pd.to_numeric(df_raw.loc[inlier_mask, 'x'], errors='coerce')
    inliers_y = pd.to_numeric(df_raw.loc[inlier_mask, 'y'], errors='coerce')

    outliers_x = pd.to_numeric(df_raw.loc[orig_indices, 'x'], errors='coerce')
    outliers_y = pd.to_numeric(df_raw.loc[orig_indices, 'y'], errors='coerce')

    plt.scatter(inliers_x, inliers_y, c='blue', s=20, alpha=0.6, label='Inliers')

    if len(orig_indices) > 0:
        plt.scatter(outliers_x, outliers_y, c='red', s=80, marker='x', linewidths=2, label='Outliers')

    plt.title(f"Outliers Detection (Density-Based)\nDataset: {dataset_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    raw_plot = os.path.join(output_dir, f"{dataset_name}_plot_raw.png")
    plt.savefig(raw_plot, dpi=150, bbox_inches='tight')
    plt.close()
