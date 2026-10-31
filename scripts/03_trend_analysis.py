import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

years = np.array([2016, 2017, 2018, 2019, 2020, 2021])
columns = ["2016-17", "2017-18", "2018-19",
           "2019-20", "2020-21", "2021-22"]

results = []

for _, row in df.iterrows():
    X = years.reshape(-1, 1)
    y = row[columns].values.astype(float)

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    results.append({
        "District": row["District"],
        "Slope": model.coef_[0],
        "Intercept": model.intercept_,
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "R2": r2_score(y, predictions)
    })

results_df = pd.DataFrame(results)

print("\nLinear Regression Results:")
print(results_df.round(3))

# Save results
results_df.to_csv("results/linear_regression_results.csv", index=False)

# Plot average trend across districts
average_connections = df[columns].mean()

plt.figure(figsize=(9, 5))
plt.plot(years, average_connections.values, marker="o")
plt.title("Average Annual PMUY Connections in West Bengal")
plt.xlabel("Year")
plt.ylabel("Average Connections")
plt.grid(True)

plt.savefig("results/average_p muy_trend.png".replace(" ", ""), dpi=300)
plt.show()

print("\nResults saved successfully!")