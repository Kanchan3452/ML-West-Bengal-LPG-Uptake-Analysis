import pandas as pd

df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

linear = pd.read_csv("results/linear_regression_results.csv")
poly = pd.read_csv("results/polynomial_regression_results.csv")
clusters = pd.read_csv("results/clustering_results.csv")

final = df.merge(linear, on="District")
final = final.merge(poly, on="District", suffixes=("_Linear", "_Polynomial"))
final = final.merge(clusters, on="District")

final.to_excel("results/final_combined_results.xlsx", index=False)

print("Final results file created successfully!")
print("Total districts:", final["District"].nunique())