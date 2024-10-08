"""Compare two time series data sets to see if one influences the other"""
import argparse
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Compares time series data in a CSV file""")
    parser.add_argument('--file', type=str, help='file to analyze')
    parser.add_argument('--target-column', type=str, help='Dependent variable')
    parser.add_argument('--feature-column', type=str, help="Independent variable")

    args = parser.parse_args()

    data = pd.read_csv(args.file)

    # clean up target data
    target_array = []
    for target in data[args.target_column]:
        if type(target) == str:
            # Remove double quotes
            cleaned_tgt = target.replace('"', "")
            if cleaned_tgt[-1] == "M":
                cleaned_tgt = cleaned_tgt.replace('M',"")
                cleaned_tgt = math.trunc(float(cleaned_tgt) * 1000000)
            elif cleaned_tgt[-1] == "B":
                cleaned_tgt = cleaned_tgt.replace('B',"")
                cleaned_tgt = math.trunc(float(cleaned_tgt) * 1000000000)
            target_array.append(cleaned_tgt)
        else:
            target_array.append(target)
    # clean up price data
    feature_array = []
    for feature in data[args.feature_column]:
        feature_array.append(feature)

    # Target Column passes ADF test
    result = adfuller(target_array)

    print(f"{args.target_column} ADF Statistic:", result[0])
    print(f"{args.target_column} p-value:", result[1])

    if result[1] >= 0.05:
        print(f"WARNING: {args.target_column} failed ADF test, this is time series data")

    # Reshape target to 2D array, as required by scikit-learn
    reshaped_target_array = np.array(target_array).reshape(-1, 1)
    feature_array = np.array(feature_array)
    model = LinearRegression()
    model.fit(reshaped_target_array, feature_array)

    # Get the slope (coefficient) and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")

    # Predict values (optional)
    predictions = model.predict(reshaped_target_array)
    print(f"Predictions: {predictions[0]}")
