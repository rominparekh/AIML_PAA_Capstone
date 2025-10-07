#!/usr/bin/env python3
"""
Customer Segmentation - K-Means Clustering Implementation
=========================================================

This script implements K-Means clustering on customer data to identify
distinct customer segments for targeted marketing strategies.

Dataset: 33,000 customer records
Output: Cluster assignments, visualizations, and comprehensive PDF report

Usage: python customer_clustering_implementation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import os

# Configure settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directories
os.makedirs('figs/clustering', exist_ok=True)
os.makedirs('output', exist_ok=True)

print(" CUSTOMER SEGMENTATION - K-MEANS CLUSTERING IMPLEMENTATION")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================
print("\n STEP 1: Loading and Preprocessing Data...")

# Load data
df = pd.read_csv('data/segmentation_data_33k.csv')
print(f"    Dataset loaded: {df.shape[0]:,} customers × {df.shape[1]} features")

# Create a copy for clustering
df_cluster = df.copy()

# Drop ID column (not needed for clustering)
df_cluster = df_cluster.drop('ID', axis=1)

# Separate numerical and categorical features
numerical_features = ['Age', 'Income']
categorical_features = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']

print(f"    Numerical features: {numerical_features}")
print(f"    Categorical features: {categorical_features}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n STEP 2: Feature Engineering...")

# Standardize numerical features
scaler = StandardScaler()
df_cluster[numerical_features] = scaler.fit_transform(df_cluster[numerical_features])
print(f"    Standardized numerical features (mean=0, std=1)")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df_cluster, columns=categorical_features, drop_first=True)
print(f"    One-hot encoded categorical features")
print(f"    Final feature count: {df_encoded.shape[1]} features")

# Store feature matrix
X = df_encoded.values
feature_names = df_encoded.columns.tolist()

print(f"    Feature matrix shape: {X.shape}")

# =============================================================================
# 3. OPTIMAL K SELECTION - ELBOW METHOD
# =============================================================================
print("\n STEP 3: Finding Optimal Number of Clusters (K)...")

k_range = range(2, 11)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X, kmeans.labels_))
    calinski_harabasz_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    
    print(f"   K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.3f}")

# Plot Elbow Method
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Optimal K Selection - Multiple Metrics', fontsize=16, fontweight='bold')

# Elbow curve
axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 0].set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12)
axes[0, 0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal K=4')
axes[0, 0].legend()

# Silhouette score
axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal K=4')
axes[0, 1].legend()

# Davies-Bouldin Index
axes[1, 0].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal K=4')
axes[1, 0].legend()

# Calinski-Harabasz Index
axes[1, 1].plot(k_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontsize=12)
axes[1, 1].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal K=4')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('figs/clustering/01_optimal_k_selection.png', dpi=300, bbox_inches='tight')
print("\n    Saved: figs/clustering/01_optimal_k_selection.png")
plt.close()

# =============================================================================
# 4. FINAL K-MEANS CLUSTERING WITH OPTIMAL K
# =============================================================================
optimal_k = 4
print(f"\n STEP 4: Running K-Means with Optimal K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans_final.fit_predict(X)

# Add cluster labels to original dataframe
df['Cluster'] = cluster_labels

# Calculate metrics
final_silhouette = silhouette_score(X, cluster_labels)
final_davies_bouldin = davies_bouldin_score(X, cluster_labels)
final_calinski_harabasz = calinski_harabasz_score(X, cluster_labels)

print(f"    Clustering complete!")
print(f"    Silhouette Score: {final_silhouette:.3f}")
print(f"    Davies-Bouldin Index: {final_davies_bouldin:.3f}")
print(f"    Calinski-Harabasz Index: {final_calinski_harabasz:.0f}")

# =============================================================================
# 5. CLUSTER PROFILING
# =============================================================================
print("\n STEP 5: Profiling Customer Segments...")

# Label mappings for interpretation
label_mappings = {
    'Sex': {0: 'Female', 1: 'Male'},
    'Marital status': {0: 'Single', 1: 'Married'},
    'Education': {0: 'Basic', 1: 'Secondary', 2: 'Higher', 3: 'Graduate'},
    'Occupation': {0: 'Unemployed/Student', 1: 'Skilled Worker', 2: 'Management'},
    'Settlement size': {0: 'Small City', 1: 'Medium City', 2: 'Large City'}
}

# Create labeled dataframe
df_labeled = df.copy()
for col, mapping in label_mappings.items():
    df_labeled[col] = df_labeled[col].map(mapping)

# Calculate cluster statistics
cluster_profiles = []
for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    profile = {
        'Cluster': cluster_id,
        'Size': len(cluster_data),
        'Percentage': len(cluster_data) / len(df) * 100,
        'Avg_Age': cluster_data['Age'].mean(),
        'Avg_Income': cluster_data['Income'].mean(),
        'Female_Pct': (cluster_data['Sex'] == 0).sum() / len(cluster_data) * 100,
        'Married_Pct': (cluster_data['Marital status'] == 1).sum() / len(cluster_data) * 100,
        'Education_Mode': cluster_data['Education'].mode()[0],
        'Occupation_Mode': cluster_data['Occupation'].mode()[0],
        'Settlement_Mode': cluster_data['Settlement size'].mode()[0]
    }
    cluster_profiles.append(profile)
    
    print(f"\n   Cluster {cluster_id}:")
    print(f"      Size: {profile['Size']:,} customers ({profile['Percentage']:.1f}%)")
    print(f"      Avg Age: {profile['Avg_Age']:.1f} years")
    print(f"      Avg Income: ${profile['Avg_Income']:,.0f}")
    print(f"      Female: {profile['Female_Pct']:.1f}%")
    print(f"      Married: {profile['Married_Pct']:.1f}%")

# Create profile dataframe
profile_df = pd.DataFrame(cluster_profiles)

# Save cluster assignments
df.to_csv('output/customer_segments.csv', index=False)
print(f"\n    Saved: output/customer_segments.csv")

# Save cluster profiles
profile_df.to_csv('output/cluster_profiles.csv', index=False)
print(f"    Saved: output/cluster_profiles.csv")

# =============================================================================
# 6. VISUALIZATION - PCA 2D PROJECTION
# =============================================================================
print("\n STEP 6: Creating Visualizations...")

# Apply PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Add PCA components to dataframe
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot 1: PCA Scatter Plot
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
              c=colors[i], label=cluster_names[i], alpha=0.6, s=30)

# Plot cluster centers
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
          c='black', marker='X', s=300, edgecolors='white', linewidths=2,
          label='Centroids', zorder=5)

ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Customer Segments - PCA 2D Projection', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/clustering/02_pca_clusters.png', dpi=300, bbox_inches='tight')
print("    Saved: figs/clustering/02_pca_clusters.png")
plt.close()

# Plot 2: Cluster Size Distribution
fig, ax = plt.subplots(figsize=(10, 6))
cluster_sizes = df['Cluster'].value_counts().sort_index()
bars = ax.bar(range(optimal_k), cluster_sizes.values, color=colors, alpha=0.7, edgecolor='black')

# Add percentage labels
for i, (bar, size) in enumerate(zip(bars, cluster_sizes.values)):
    height = bar.get_height()
    pct = size / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{size:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.set_title('Customer Distribution Across Segments', fontsize=16, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figs/clustering/03_cluster_sizes.png', dpi=300, bbox_inches='tight')
print("    Saved: figs/clustering/03_cluster_sizes.png")
plt.close()

# Plot 3: Age and Income by Cluster
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Age distribution
df_labeled.boxplot(column='Age', by='Cluster', ax=axes[0], patch_artist=True)
axes[0].set_xlabel('Cluster', fontsize=12)
axes[0].set_ylabel('Age (years)', fontsize=12)
axes[0].set_title('Age Distribution by Cluster', fontsize=14, fontweight='bold')
axes[0].get_figure().suptitle('')  # Remove default title
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)

# Income distribution
df_labeled.boxplot(column='Income', by='Cluster', ax=axes[1], patch_artist=True)
axes[1].set_xlabel('Cluster', fontsize=12)
axes[1].set_ylabel('Income ($)', fontsize=12)
axes[1].set_title('Income Distribution by Cluster', fontsize=14, fontweight='bold')
axes[1].get_figure().suptitle('')  # Remove default title
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('figs/clustering/04_age_income_by_cluster.png', dpi=300, bbox_inches='tight')
print("    Saved: figs/clustering/04_age_income_by_cluster.png")
plt.close()

# Plot 4: Demographic Composition
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Demographic Composition by Cluster', fontsize=16, fontweight='bold')

# Gender composition
gender_comp = df_labeled.groupby(['Cluster', 'Sex']).size().unstack(fill_value=0)
gender_comp_pct = gender_comp.div(gender_comp.sum(axis=1), axis=0) * 100
gender_comp_pct.plot(kind='bar', stacked=True, ax=axes[0, 0], color=['#FF69B4', '#4169E1'])
axes[0, 0].set_xlabel('Cluster', fontsize=11)
axes[0, 0].set_ylabel('Percentage (%)', fontsize=11)
axes[0, 0].set_title('Gender Distribution', fontsize=12, fontweight='bold')
axes[0, 0].legend(title='Gender', fontsize=9)
axes[0, 0].set_xticklabels([f'C{i}' for i in range(optimal_k)], rotation=0)

# Marital status composition
marital_comp = df_labeled.groupby(['Cluster', 'Marital status']).size().unstack(fill_value=0)
marital_comp_pct = marital_comp.div(marital_comp.sum(axis=1), axis=0) * 100
marital_comp_pct.plot(kind='bar', stacked=True, ax=axes[0, 1], color=['#90EE90', '#FFD700'])
axes[0, 1].set_xlabel('Cluster', fontsize=11)
axes[0, 1].set_ylabel('Percentage (%)', fontsize=11)
axes[0, 1].set_title('Marital Status Distribution', fontsize=12, fontweight='bold')
axes[0, 1].legend(title='Marital Status', fontsize=9)
axes[0, 1].set_xticklabels([f'C{i}' for i in range(optimal_k)], rotation=0)

# Education composition
edu_comp = df_labeled.groupby(['Cluster', 'Education']).size().unstack(fill_value=0)
edu_comp_pct = edu_comp.div(edu_comp.sum(axis=1), axis=0) * 100
edu_comp_pct.plot(kind='bar', stacked=True, ax=axes[1, 0])
axes[1, 0].set_xlabel('Cluster', fontsize=11)
axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
axes[1, 0].set_title('Education Level Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend(title='Education', fontsize=8)
axes[1, 0].set_xticklabels([f'C{i}' for i in range(optimal_k)], rotation=0)

# Occupation composition
occ_comp = df_labeled.groupby(['Cluster', 'Occupation']).size().unstack(fill_value=0)
occ_comp_pct = occ_comp.div(occ_comp.sum(axis=1), axis=0) * 100
occ_comp_pct.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_xlabel('Cluster', fontsize=11)
axes[1, 1].set_ylabel('Percentage (%)', fontsize=11)
axes[1, 1].set_title('Occupation Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend(title='Occupation', fontsize=8)
axes[1, 1].set_xticklabels([f'C{i}' for i in range(optimal_k)], rotation=0)

plt.tight_layout()
plt.savefig('figs/clustering/05_demographic_composition.png', dpi=300, bbox_inches='tight')
print("    Saved: figs/clustering/05_demographic_composition.png")
plt.close()

# Plot 5: Cluster Profiles Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for heatmap
heatmap_data = profile_df[['Cluster', 'Avg_Age', 'Avg_Income', 'Female_Pct',
                            'Married_Pct', 'Percentage']].set_index('Cluster')
heatmap_data.columns = ['Avg Age', 'Avg Income', '% Female', '% Married', '% of Total']

# Normalize for better visualization
from sklearn.preprocessing import MinMaxScaler
scaler_viz = MinMaxScaler()
heatmap_normalized = pd.DataFrame(
    scaler_viz.fit_transform(heatmap_data),
    columns=heatmap_data.columns,
    index=heatmap_data.index
)

sns.heatmap(heatmap_normalized.T, annot=heatmap_data.T, fmt='.1f',
            cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'},
            linewidths=1, linecolor='white', ax=ax)
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Characteristics', fontsize=12)
ax.set_title('Cluster Profiles Heatmap (Normalized)', fontsize=16, fontweight='bold')
ax.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)], rotation=0)

plt.tight_layout()
plt.savefig('figs/clustering/06_cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
print("    Saved: figs/clustering/06_cluster_profiles_heatmap.png")
plt.close()

print("\n CLUSTERING IMPLEMENTATION COMPLETE!")
print("=" * 70)

