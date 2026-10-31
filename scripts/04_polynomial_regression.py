import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

years = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)

columns = [
    "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22"
]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(years)

results = []

for _, row in df.iterrows():
    y = row[columns].values.astype(float)

    model = LinearRegression()
    model.fit(X_poly, y)

    predictions = model.predict(X_poly)

    results.append({
        "District": row["District"],
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "R2": r2_score(y, predictions)
    })

results_df = pd.DataFrame(results)

print("\nPolynomial Regression Results:")
print(results_df.round(3))

results_df.to_csv(
    "results/polynomial_regression_results.csv",
    index=False
)

print("\nResults saved successfully!")