import pandas as pd

df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

print("Duplicate districts:", df["District"].duplicated().sum())

print("\nMissing NFHS-5 values:")
print(df[df["NFHS5_Clean_Fuel_Percent"].isnull()]["District"])

complete_years = [
    "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22"
]

all_years = complete_years + ["2022-23_Apr-Feb"]

df["Total_PMUY_Connections"] = df[all_years].sum(axis=1)

df["Average_Annual_Connections"] = df[complete_years].mean(axis=1)

df["Total_Complete_Year_Connections"] = df[complete_years].sum(axis=1)

print("\nProcessed Dataset:")
print(df[[
    "District",
    "Total_PMUY_Connections",
    "Average_Annual_Connections",
    "Total_Complete_Year_Connections",
    "NFHS5_Clean_Fuel_Percent"
]].head(10))

df.to_csv("results/processed_dataset.csv", index=False)

print("\nPreprocessing completed!")