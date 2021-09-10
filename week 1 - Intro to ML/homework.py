import numpy as np
import pandas as pd
from statistics import mean

# 1. What's the version of NumPy that you installed?
np.__version__

# 2. What's the version of Pandas?
pd.__version__

# 3. What's the average price of BMW cars in the dataset?
df = pd.read_csv("data.csv")
df.head(5)

bmw_df = df[df["Make"] == "BMW"]
print("The average price of BMW cars in the dataset is: ", np.average(bmw_df['MSRP']))

# 4. Select a subset of cars after year 2015 (inclusive, i.e. 2015 and after). How many of them have missing values for
# Engine HP?

df_extracted = df[df["Year"] >= 2015]
df_extracted["Engine HP"].isnull().sum()

# 5. Calculate the average "Engine HP" in the dataset.
# Use the fillna method and to fill the missing values in "Engine HP" with the mean value from the previous step.
# Now, calcualte the average of "Engine HP" again.
# Has it changed?

print("The average engine HP before transformation in the dataset is: ", df_extracted["Engine HP"].mean(axis=0))

mean_before = df_extracted["Engine HP"].mean(axis=0)

df_extracted['Engine HP'].fillna(value=2, inplace=True)
mean_after = df_extracted["Engine HP"].mean(axis=0)
print("The average engine HP after transformation in the dataset is: ", mean_after)

# Question 6
# Select all the "Rolls-Royce" cars from the dataset.
# Select only columns "Engine HP", "Engine Cylinders", "highway MPG".
# Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 7 rows).
# Get the underlying NumPy array. Let's call it X.
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the
# result XTX.
# Invert XTX.
# What's the sum of all the elements of the result?
rr_df = df[df["Make"] == "Rolls-Royce"]

df_subset = rr_df[["Engine HP", "Engine Cylinders", "highway MPG"]]
df_subset = df_subset.drop_duplicates()

X = np.array(df_subset)
XTX = X.T
XTX_inverted = np.linalg.inv(XTX)    # Problems as last 2 dimensions of the array must be square for this operation

# Question 7
# Create an array y with values [1000, 1100, 900, 1200, 1000, 850, 1300].
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# What's the value of the first element of w?.
y = [1000, 1100, 900, 1200, 1000, 850, 1300]

