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

print("The average engine HP before transformation in the dataset is: ", np.average(df_extracted['Engine HP']))
mean(df_extracted["Engine HP"])
mean([1, 2, 3])