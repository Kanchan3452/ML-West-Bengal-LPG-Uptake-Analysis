import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

# Remove districts with missing NFHS-5 values
nfhs = df.dropna(subset=["NFHS5_Clean_Fuel_Percent"]).copy()

# Display district-wise clean fuel usage
nfhs = nfhs.sort_values("NFHS5_Clean_Fuel_Percent", ascending=False)

print("\nDistrict-wise NFHS-5 Clean Fuel Usage:")
print(nfhs[["District", "NFHS5_Clean_Fuel_Percent"]])

print("\nAverage Clean Fuel Usage:")
print(round(nfhs["NFHS5_Clean_Fuel_Percent"].mean(), 2), "%")

# Bar chart
plt.figure(figsize=(12, 6))
plt.bar(nfhs["District"], nfhs["NFHS5_Clean_Fuel_Percent"])
plt.title("District-wise Clean Fuel Usage in West Bengal (NFHS-5)")
plt.xlabel("District")
plt.ylabel("Households Using Clean Fuel (%)")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("results/nfhs5_clean_fuel.png", dpi=300)
plt.show()

# Save results
nfhs[["District", "NFHS5_Clean_Fuel_Percent"]].to_csv(
    "results/nfhs5_analysis.csv", index=False
)

print("\nNFHS-5 analysis completed!")