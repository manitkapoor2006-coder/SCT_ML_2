# ============================================================
#  K-MEANS CUSTOMER SEGMENTATION
#  SkillCraft Technology — Task 02
#  Author- MANIT
# ============================================================


import urllib.request
import os

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/mall_customers.csv"
filename = "Mall_Customers.csv"

if not os.path.exists(filename):
    try:
        urllib.request.urlretrieve(url, filename)
        print("✅ Dataset downloaded successfully!")
    except:
        # Fallback — generate synthetic data if download fails
        print("⚠️ Download failed. Generating synthetic dataset...")
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        n = 200
        clusters_data = {
            'CustomerID': range(1, n+1),
            'Genre': np.random.choice(['Male','Female'], n),
            'Age': np.random.randint(18, 70, n),
            'Annual Income (k$)': np.concatenate([
                np.random.normal(80, 8, 40),   # High income
                np.random.normal(80, 8, 40),   # High income
                np.random.normal(50, 8, 40),   # Mid income
                np.random.normal(25, 8, 40),   # Low income
                np.random.normal(25, 8, 40),   # Low income
            ]).clip(15, 140).astype(int),
            'Spending Score (1-100)': np.concatenate([
                np.random.normal(82, 8, 40),   # High spend
                np.random.normal(18, 8, 40),   # Low spend
                np.random.normal(50, 10, 40),  # Mid spend
                np.random.normal(79, 8, 40),   # High spend
                np.random.normal(20, 8, 40),   # Low spend
            ]).clip(1, 100).astype(int)
        }
        df_syn = pd.DataFrame(clusters_data)
        df_syn.to_csv(filename, index=False)
        print("✅ Synthetic dataset created (200 customers, 5 clusters)")
else:
    print("✅ Dataset already exists!")


# ── STEP 2: Import Libraries ─────────────────────────────────────────
# All libraries are pre-installed in Google Colab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries loaded successfully!")


# ── STEP 3: Load Data & Basic Info ───────────────────────────────────

df = pd.read_csv(filename)
df.columns = df.columns.str.strip()

print("\n" + "="*50)
print("  DATASET INFO")
print("="*50)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic Statistics:")
print(df.describe().round(2))
print(f"\nMissing values: {df.isnull().sum().sum()}")


# ── STEP 4: EDA Plots ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')

axes[0].hist(df['Age'], bins=20, color='#58a6ff', edgecolor='#0d1117', linewidth=0.8)
axes[0].set_title('Age Distribution', color='white', fontsize=12, pad=10)
axes[0].set_xlabel('Age', color='#8b949e')
axes[0].tick_params(colors='#8b949e')
for spine in axes[0].spines.values(): spine.set_color('#30363d')

axes[1].hist(df['Annual Income (k$)'], bins=20, color='#3fb950', edgecolor='#0d1117', linewidth=0.8)
axes[1].set_title('Annual Income (k$)', color='white', fontsize=12, pad=10)
axes[1].set_xlabel('Income (k$)', color='#8b949e')
axes[1].tick_params(colors='#8b949e')
for spine in axes[1].spines.values(): spine.set_color('#30363d')

axes[2].hist(df['Spending Score (1-100)'], bins=20, color='#f78166', edgecolor='#0d1117', linewidth=0.8)
axes[2].set_title('Spending Score', color='white', fontsize=12, pad=10)
axes[2].set_xlabel('Score (1-100)', color='#8b949e')
axes[2].tick_params(colors='#8b949e')
for spine in axes[2].spines.values(): spine.set_color('#30363d')

fig.suptitle('Exploratory Data Analysis — Mall Customers',
             color='white', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ EDA plot saved: eda_plots.png")


# ── STEP 5: Select Features for Clustering ───────────────────────────

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
print(f"\nClustering features: Annual Income & Spending Score")
print(f"Total data points: {len(X)}")


# ── STEP 6: Elbow Method + Silhouette Score ──────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
sil_scores = []
K_range = range(2, 11)

print("\n⏳ Searching for optimal K...")
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K={k}  |  Inertia: {km.inertia_:.1f}  |  Silhouette: {sil_scores[-1]:.4f}")

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"\n Best K based on Silhouette Score = {best_k}")

# Plot Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')

axes[0].plot(K_range, inertias, 'o-', color='#58a6ff', linewidth=2.5,
             markersize=8, markerfacecolor='white', markeredgecolor='#58a6ff')
axes[0].set_title('Elbow Method — WCSS (Inertia)', color='white', fontsize=12, pad=10)
axes[0].set_xlabel('Number of Clusters (K)', color='#8b949e')
axes[0].set_ylabel('Inertia', color='#8b949e')
axes[0].tick_params(colors='#8b949e')
axes[0].grid(True, alpha=0.15, color='#30363d')
for spine in axes[0].spines.values(): spine.set_color('#30363d')

axes[1].plot(K_range, sil_scores, 'o-', color='#f78166', linewidth=2.5,
             markersize=8, markerfacecolor='white', markeredgecolor='#f78166')
axes[1].axvline(x=best_k, color='#3fb950', linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Best K = {best_k}')
axes[1].set_title('Silhouette Score (higher = better)', color='white', fontsize=12, pad=10)
axes[1].set_xlabel('Number of Clusters (K)', color='#8b949e')
axes[1].set_ylabel('Silhouette Score', color='#8b949e')
axes[1].tick_params(colors='#8b949e')
axes[1].grid(True, alpha=0.15, color='#30363d')
axes[1].legend(facecolor='#30363d', edgecolor='#58a6ff', labelcolor='white')
for spine in axes[1].spines.values(): spine.set_color('#30363d')

fig.suptitle('Finding Optimal K', color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ Elbow plot saved: elbow_silhouette.png")


# ── STEP 7: Final K-Means Clustering ────────────────────────────────

K = 5   # Change this value if needed (or use best_k for automatic selection)
# K = best_k  # Uncomment this line to use the best K automatically

km_final = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
labels = km_final.fit_predict(X)

sil = silhouette_score(X, labels)
print(f"\n{'='*50}")
print(f"  K-MEANS RESULTS (K={K})")
print(f"{'='*50}")
print(f"Inertia (WCSS)  : {km_final.inertia_:.2f}")
print(f"Silhouette Score: {sil:.4f}  (range: 0 to 1, higher is better)")


# ── STEP 8: Cluster Scatter Plot ─────────────────────────────────────

COLORS = ['#f78166', '#58a6ff', '#3fb950', '#d2a8ff', '#ffa657',
          '#79c0ff', '#56d364', '#ff7b72', '#ffa657']

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

for i in range(K):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1],
               color=COLORS[i], label=f'Cluster {i+1}',
               alpha=0.8, s=70, edgecolors='#0d1117', linewidths=0.5)

# Plot Centroids
ax.scatter(km_final.cluster_centers_[:, 0],
           km_final.cluster_centers_[:, 1],
           color='white', marker='*', s=300, zorder=10,
           label='Centroids', edgecolors='black', linewidths=0.8)

ax.set_title(f'K-Means Clustering — K={K}\nAnnual Income vs Spending Score',
             color='white', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Annual Income (k$)', color='#8b949e', fontsize=11)
ax.set_ylabel('Spending Score (1-100)', color='#8b949e', fontsize=11)
ax.tick_params(colors='#8b949e')
ax.grid(True, alpha=0.1, color='#30363d')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white',
          loc='upper left', fontsize=9)
for spine in ax.spines.values(): spine.set_color('#30363d')

plt.tight_layout()
plt.savefig(f'clusters_k{K}.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"✅ Cluster plot saved: clusters_k{K}.png")


# ── STEP 9: Cluster Analysis Report ─────────────────────────────────

df['Cluster'] = labels + 1

CLUSTER_NAMES = {
    1: 'High Income, High Spenders   (VIP Target Customers)',
    2: 'High Income, Low Spenders    (Careful Savers)',
    3: 'Average Income, Average Spend (Middle Segment)',
    4: 'Low Income, High Spenders    (Impulsive Buyers)',
    5: 'Low Income, Low Spenders     (Budget Conscious)',
}

print("\n" + "="*60)
print("  CLUSTER ANALYSIS REPORT")
print("="*60)

summary = df.groupby('Cluster').agg(
    Count=('Annual Income (k$)', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Income_k=('Annual Income (k$)', 'mean'),
    Avg_Spending_Score=('Spending Score (1-100)', 'mean')
).round(1)

print(summary.to_string())

print("\nCluster Interpretation (K=5 default):")
for k, name in CLUSTER_NAMES.items():
    if k <= K:
        row = summary.loc[k] if k in summary.index else None
        if row is not None:
            print(f"\n  Cluster {k}: {name}")
            print(f"    -> {int(row['Count'])} customers | "
                  f"Avg Income: {row['Avg_Income_k']}k | "
                  f"Avg Spending Score: {row['Avg_Spending_Score']}")

# Save clustered data
df.to_csv('customers_clustered.csv', index=False)
print("\n✅ Clustered data saved: customers_clustered.csv")


# ── STEP 10: Summary Bar Chart ───────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')

x = np.arange(K)
cluster_labels_x = [f'C{i+1}' for i in range(K)]

bars1 = axes[0].bar(x, summary['Avg_Income_k'], color=COLORS[:K],
                    edgecolor='#0d1117', linewidth=0.8)
axes[0].set_title('Avg Annual Income per Cluster', color='white', fontsize=12, pad=10)
axes[0].set_xlabel('Cluster', color='#8b949e')
axes[0].set_ylabel('Income (k$)', color='#8b949e')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cluster_labels_x)
axes[0].tick_params(colors='#8b949e')
axes[0].grid(True, alpha=0.1, axis='y', color='#30363d')
for spine in axes[0].spines.values(): spine.set_color('#30363d')
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.0f}k', ha='center', va='bottom',
                color='white', fontsize=9)

bars2 = axes[1].bar(x, summary['Avg_Spending_Score'], color=COLORS[:K],
                    edgecolor='#0d1117', linewidth=0.8)
axes[1].set_title('Avg Spending Score per Cluster', color='white', fontsize=12, pad=10)
axes[1].set_xlabel('Cluster', color='#8b949e')
axes[1].set_ylabel('Spending Score', color='#8b949e')
axes[1].set_xticks(x)
axes[1].set_xticklabels(cluster_labels_x)
axes[1].tick_params(colors='#8b949e')
axes[1].grid(True, alpha=0.1, axis='y', color='#30363d')
for spine in axes[1].spines.values(): spine.set_color('#30363d')
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.0f}', ha='center', va='bottom',
                color='white', fontsize=9)

fig.suptitle('Cluster Comparison Summary', color='white',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_summary.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ Summary chart saved: cluster_summary.png")


# ── DONE ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"  Dataset      : Mall_Customers.csv ({df.shape[0]} customers)")
print(f"  Clusters     : K = {K}")
print(f"  Silhouette   : {sil:.4f}")
print(f"  Inertia      : {km_final.inertia_:.2f}")
print("\n  Files saved:")
print("     eda_plots.png")
print("     elbow_silhouette.png")
print(f"     clusters_k{K}.png")
print("     cluster_summary.png")
print("     customers_clustered.csv")
print("\n  To download files: open the Files tab on the left in Colab")
print("="*60)
