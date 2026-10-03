import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# Input data
raw_temp = [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23, 22, 21, 23, 24, 25]
df = pd.DataFrame({"Hour": range(1, 16), "Temp_raw": raw_temp})

# (a) Display full descriptive statistics BEFORE cleaning
valid_temp = df["Temp_raw"].dropna()
print("Stats BEFORE cleaning:")
print(f"Mean: {valid_temp.mean():.2f} Std: {valid_temp.std():.2f}")
print(f"Skewness: {stats.skew(valid_temp):.2f} Kurtosis: {stats.kurtosis(valid_temp):.2f}")

# (b) Fill missing readings using linear interpolation
df["Temp_clean"] = df["Temp_raw"].interpolate(method="linear")
print(f"\nAfter interpolation, missing values: {df['Temp_clean'].isnull().sum()}")

# (c) Detect faulty outlier using Z-score method (threshold=2)
temp_z_initial = zscore(df["Temp_clean"])
outlier_mask = np.abs(temp_z_initial) > 2

outliers = df[outlier_mask]
for idx, row in outliers.iterrows():
    print(f"Z-score(>2) outlier detected at Hour: {int(row['Hour'])} (value: {row['Temp_clean']})")

# Replace only the flagged outlier with the column median
median_val = df["Temp_clean"].median()
df.loc[outlier_mask, "Temp_clean"] = median_val
print(f"Replaced with column median: {median_val}")

# (d) Apply Z-score standardization to the final cleaned column
df["Temp_Zscore"] = zscore(df["Temp_clean"])

print("\nBefore/After comparison (first 8 rows):")
print(df.head(8))

print("\nStats AFTER cleaning:")
print(f"Mean: {df['Temp_clean'].mean():.2f} Std: {df['Temp_clean'].std():.2f}")
