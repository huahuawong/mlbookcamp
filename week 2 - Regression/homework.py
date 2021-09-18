# The goal of this homework is to create a regression model for prediction apartment prices (column 'price').

# EDA
# Load the data.
# Look at the price variable. Does it have a long tail?
# Features
import pandas as pd
import numpy as np

df = pd.read_csv("AB_NYC_2019.csv")
df.head(5)
df.info()
df.describe()

df['price'].hist(bins=50, color='steelblue', edgecolor='black', linewidth = 1.0, grid = False, alpha = 0.65)
# From the histogram plot, we can see that it has a long tail, in fact, it is rightly skewed

# For the rest of the homework, you'll need to use only these columns:
# 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
# 'calculated_host_listings_count', 'availability_365'
# Select only them.
df_extracted = df[['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365']]

# Question 1
# Find a feature with missing values. How many missing values does it have?
df_extracted.info()
# We cam see that "reviews_per_month" has missing values
df_extracted['reviews_per_month'].isnull().sum()

# 10052 null entries


# Question 2
# What's the median (50% percentile) for variable 'minimum_nights'?
stats = df_extracted.describe()
# We can see that the median is 3, it can also be checked using the command below:
df_extracted['minimum_nights'].median()

# Split the data
# Shuffle the initial dataset, use seed 42.
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
# Make sure that the target value ('price') is not in your dataframe.
# Apply the log transformation to the price variable using the np.log1p() function.
# Question 3
df_copy = df_extracted.copy()
# df_copy = df_copy.drop(['price'], axis=1)

# train, validate, test = np.split(df_copy.sample(frac=1, random_state=42),
#                         [int(.6 * len(df_copy)), int(.8 * len(df_copy))])

n = len(df_copy)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df_train = df_copy.iloc[n_train:]
df_val = df_copy.iloc[n_train:n_train+n_val]
df_test = df_copy.iloc[n_train+n_val:]

idx = np.arange(n)

np.random.seed(42)
np.random.shuffle(idx)

df_train = df_copy.iloc[idx[:n_train]]
df_val = df_copy.iloc[idx[n_train:n_train+n_val]]
df_test = df_copy.iloc[idx[n_train+n_val:]]
df_train.head()

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

len(y_train)

# We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
# Try both options. For each, train a linear regression model without regularization using the code from the lesssons.
# For computing the mean, use the training only!
# Compare the RMSE of each option.
# Round the RMSE scores to 2 decimal digits using round(score, 2)
# Which option gives better RMSE?

# Option 1, filling it with 0

df_train.info()
df_train = df_train.fillna(0)


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


X_train = df_train.values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


print(f'The RMSE with zero imputation is: {round(rmse(y_train, y_pred), 2)}')
# 0.49


# Option 2, filling it with mean
df_train.info()
mean_value = df_train['reviews_per_month'].mean()
df_train['reviews_per_month'].fillna(value=mean_value, inplace=True)

X_train = df_train.values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)
rmse(y_train, y_pred)
print(f'The RMSE with mean imputation is: {round(rmse(y_train, y_pred), 2)}')
# 0.49
# Similar RMSE

# Question 4
# Now let's train a regularized linear regression.
# For this question, fill the NAs with 0.
df_train.info()
df_train = df_train.fillna(0)
df_val = df_val.fillna(0)

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

# Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
    X_train = df_train.values
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = df_val.values
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(r, w0, score)
# If there are multiple options, select the smallest r.
# when r equals to 0


# Question 5
# We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# Fill the missing values with 0 and train a model without regularization.
# For each seed, collect the RMSE scores.
# What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3))
# Note: Standard deviation shows how different the values are. If it's low, then all values are approximately the same.
# If it's high, the values are different. If standard deviation of scores is low, then our model is stable.
rmse_arr = []
for num_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    df_train = df_copy.iloc[n_train:]
    df_val = df_copy.iloc[n_train:n_train+n_val]
    df_test = df_copy.iloc[n_train+n_val:]

    idx = np.arange(n)

    np.random.seed(num_seed)
    np.random.shuffle(idx)

    df_train = df_copy.iloc[idx[:n_train]]
    df_val = df_copy.iloc[idx[n_train:n_train+n_val]]
    df_test = df_copy.iloc[idx[n_train+n_val:]]
    # df_train.head()

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)

    # df_train.info()
    df_train = df_train.fillna(0)

    X_train = df_train.values
    w0, w = train_linear_regression(X_train, y_train)
    y_pred = w0 + X_train.dot(w)

    print(f'The RMSE with random seed of {num_seed} is: {round(rmse(y_train, y_pred), 2)}')
    rmse_arr.append(round(rmse(y_train, y_pred), 2))

std_dev_rmse = round(np.std(rmse_arr), 3)
print(f'The standard deviation of RMSE is: {std_dev_rmse}, which is low as the values are similar')

# Question 6
# Split the dataset like previously, use seed 9.
# Combine train and validation datasets.
# Train a model with r=0.001.
# What's the RMSE on test dataset?
# Submit the results
df_train = df_copy.iloc[n_train:]
df_val = df_copy.iloc[n_train:n_train + n_val]
df_test = df_copy.iloc[n_train + n_val:]

idx = np.arange(n)

np.random.seed(9)
np.random.shuffle(idx)

df_train = df_copy.iloc[idx[:n_train]]
df_val = df_copy.iloc[idx[n_train:n_train + n_val]]
df_test = df_copy.iloc[idx[n_train + n_val:]]
# df_train.head()

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

df_x_combined = pd.concat([df_train, df_val])
df_y_combined = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_val)])

# df_train.info()
df_train = df_x_combined.fillna(0)

X_train = df_train.values
y_train = df_y_combined.values

w0, w = train_linear_regression_reg(X_train, y_train, r=0.001)

df_test = df_test.fillna(0)

X_test = df_test.values
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)

print(r, w0, score)
# 0.866
