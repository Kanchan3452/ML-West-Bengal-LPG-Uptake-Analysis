import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("data/West_Bengal_PMUY_NFHS5_Combined.xlsx")

columns = [
    "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22"
]

X = np.arange(6).reshape(-1, 1)
loo = LeaveOneOut()

results = []

for _, row in df.iterrows():

    y = row[columns].values.astype(float)

    linear = LinearRegression()
    polynomial = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )

    linear_pred = cross_val_predict(linear, X, y, cv=loo)
    poly_pred = cross_val_predict(polynomial, X, y, cv=loo)

    linear_rmse = np.sqrt(mean_squared_error(y, linear_pred))
    poly_rmse = np.sqrt(mean_squared_error(y, poly_pred))

    results.append({
        "District": row["District"],
        "Linear_RMSE": linear_rmse,
        "Polynomial_RMSE": poly_rmse,
        "Linear_R2": r2_score(y, linear_pred),
        "Polynomial_R2": r2_score(y, poly_pred),
        "Better_Model": (
            "Polynomial" if poly_rmse < linear_rmse else "Linear"
        )
    })

comparison = pd.DataFrame(results)

print("\nCross-Validation Model Comparison:")
print(comparison.round(3))

print("\nAverage Performance:")
print(comparison.drop(columns="District").select_dtypes(
    include="number"
).mean().round(3))

print("\nOverall Model Comparison:")
print(comparison["Better_Model"].value_counts())

comparison.to_csv("results/model_comparison.csv", index=False)

print("\nComparison saved successfully!")