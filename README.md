# K-Means Customer Segmentation 🎯

An end-to-end Machine Learning project to segment retail mall customers based on their purchasing behavior and financial profiles. This project was developed as part of **Task 02** for the **SkillCraft Technology** internship.

The script applies unsupervised learning (**K-Means Clustering**) using key target metrics—**Annual Income** and **Spending Score**—to assist marketing teams in deploying targeted, data-driven customer campaigns.

---

## 🚀 Features

* **Smart Data Ingestion:** Automatically downloads the live dataset from a remote source or gracefully falls back to generating a distribution-accurate synthetic dataset if working offline.
* **Optimal K Selection:** Evaluates the ideal number of clusters using both the mathematical **Elbow Method (WCSS)** and **Silhouette Score** optimization metrics.
* **Custom Dark-Themed Visualizations:** High-quality, presentation-ready plots matching modern dark-mode IDE environments (EDA, Cluster Geometry, and Performance evaluation).
* **Business-Oriented Labeling:** Automatically profiles and interprets clusters into actionable buyer personas (e.g., *VIP Target Customers*, *Careful Savers*).

---

## 📊 Customer Person Mapping (K=5)

The algorithm partitions consumers into 5 highly distinct tactical segments:

| Cluster ID | Segment Name | Profile Description |
| :--- | :--- | :--- |
| **Cluster 1** | 💎 VIP Target Customers | High Income, High Spending Score |
| **Cluster 2** | 🛡️ Careful Savers | High Income, Low Spending Score |
| **Cluster 3** | ⚖️ Middle Segment | Average Income, Average Spending Score |
| **Cluster 4** | 🛍️ Impulsive Buyers | Low Income, High Spending Score |
| **Cluster 5** | 📉 Budget Conscious | Low Income, Low Spending Score |

---

## 📂 Project Structure

```text
├── Mall_Customers.csv         # Downloaded or fallback synthetic dataset
├── main_segmentation.py       # Core Python execution script
├── eda_plots.png             # Feature distribution visualizations
├── elbow_silhouette.png       # K-selection evaluation graphs
├── clusters_k5.png            # Final 2D spatial scatter plot of clusters
├── cluster_summary.png        # Bar chart comparison reports
├── customers_clustered.csv    # Final dataset appended with assigned Cluster IDs
└── README.md                  # Project documentation
