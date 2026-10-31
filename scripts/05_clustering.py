import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Load dataset
df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

columns = [
    "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22"
]

# Scale annual PMUY data
X = df[columns].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the best number of clusters
scores = []

for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    scores.append((k, score))
    print(f"K = {k}, Silhouette Score = {score:.3f}")

best_k = max(scores, key=lambda x: x[1])[0]
print("\nBest number of clusters:", best_k)

# K-Means clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=best_k)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

print("\nDistrict Cluster Assignments:")
print(df[["District", "KMeans_Cluster", "Hierarchical_Cluster"]])

# Calculate silhouette scores
print("\nK-Means Silhouette Score:",
      round(silhouette_score(X_scaled, df["KMeans_Cluster"]), 3))

print("Hierarchical Silhouette Score:",
      round(silhouette_score(X_scaled, df["Hierarchical_Cluster"]), 3))

# Save results
df[["District", "KMeans_Cluster", "Hierarchical_Cluster"]].to_csv(
    "results/clustering_results.csv", index=False
)

# Create dendrogram
linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked, labels=df["District"].values, leaf_rotation=90)
plt.title("Hierarchical Clustering of West Bengal Districts")
plt.xlabel("Districts")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("results/hierarchical_dendrogram.png", dpi=300)
plt.show()

print("\nClustering results saved successfully!")