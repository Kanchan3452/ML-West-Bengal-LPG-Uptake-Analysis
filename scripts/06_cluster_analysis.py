import pandas as pd

df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

results = pd.read_csv("results/clustering_results.csv")

df = df.merge(results, on="District")

columns = [
    "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22"
]

print("\nK-Means Cluster Summary:")
print(df.groupby("KMeans_Cluster")[columns].mean().round(2))

print("\nDistricts in Each K-Means Cluster:")
for cluster in sorted(df["KMeans_Cluster"].unique()):
    print(f"\nCluster {cluster}:")
    print(df[df["KMeans_Cluster"] == cluster]["District"].tolist())

print("\nCluster-wise NFHS-5 Clean Fuel Average:")
print(df.groupby("KMeans_Cluster")["NFHS5_Clean_Fuel_Percent"].mean().round(2))

print("\nAnalysis completed!")