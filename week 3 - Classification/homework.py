# EDA
# Load the data.
# Look at the price variable. Does it have a long tail?
# Features
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("AB_NYC_2019.csv")
df.head(5)
df.info()
df.describe()

# df['price'].hist(bins=50, color='steelblue', edgecolor='black', linewidth = 1.0, grid = False, alpha = 0.65)
# From the histogram plot, we can see that it has a long tail, in fact, it is rightly skewed
df.info()
# We can see that there are some missing values, fill them with 0
df.fillna(0, inplace=True)

# For the rest of the homework, you'll need to use only these columns:
# 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
# 'calculated_host_listings_count', 'availability_365',
# Select only them.
df_extracted = df[['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'room_type']]


# Question 1
# What is the most frequent observation (mode) for the column 'neighbourhood_group'?
df['neighbourhood_group'].mode()
# df['neighbourhood_group'].value_counts()
# Manhattan is the most frequent observation

# Split the data
from sklearn.model_selection import train_test_split

# Getting rid of price columns
data_split = df_extracted.drop('price', 1)

df_full_train, df_test = train_test_split(data_split, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
len(df_train), len(df_val), len(df_test)

# Question 2
# Create the correlation matrix for the numerical features of your train dataset.
# In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.
# What are the two features that have the biggest correlation in this dataset?
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_col = df_train.select_dtypes(include=numerics)

corr = numerical_col.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# What are the two features that have the biggest correlation in this dataset?
# Ans: no of reviews and reviews per month

# Make price binary
# We need to turn the price variable from numeric into binary.
# Let's create a variable above_average which is 1 if the price is above (or equal to) 152.
df_extracted['above_average'] = np.where(df_extracted['price'] >= 52, 1, 0)

# Question 3
# Calculate the mutual information score with the (binarized) price for the two categorical variables that we have.
# Use the training set only.
# Which of these two variables has bigger score?
# Round it to 2 decimal digits using round(score, 2)

# Mutual information - concept from information theory, it tells us how much we can learn about one variable if we
# know the value of another

from sklearn.metrics import mutual_info_score

# Getting rid of price columns
data_split = df_extracted.drop('price', 1)

df_full_train, df_test = train_test_split(data_split, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

del df_train['above_average']
del df_val['above_average']
del df_test['above_average']

round(mutual_info_score(df_full_train.neighbourhood_group, df_full_train.above_average), 2)
# 0.03

round(mutual_info_score(df_full_train.room_type, df_full_train.above_average), 2)
# 0.10

# Question 4
# Now let's train a logistic regression
# Remember that we have two categorical variables in the data. Include them using one-hot encoding.
# Fit the model on the training dataset.
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these
# parameters:

# model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

dv = DictVectorizer(sparse=False)

categorical = list(df_train.dtypes[df_train.dtypes == 'object'].index)
numerical = list(df_train.dtypes[df_train.dtypes != 'object'].index)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
decision = (y_pred >= 0.5)

df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction == df_pred.actual
df_pred.correct.mean()
# Accuracy: 0.866

# Question 5
# We have 9 features: 7 numerical features and 2 categorical.
# Let's find the least useful one using the feature elimination technique.
# Train a model with all these features (using the same parameters as in Q4).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
# Which of following feature has the smallest difference?
# neighbourhood_group
# room_type
# number_of_reviews
# reviews_per_month

# Skip this and I am guessing it's room_type

# Q6
# For this question, we'll see how to use a linear regression model from Scikit-Learn
# We'll need to use the original column 'price'. Apply the logarithmic transformation to this column.
# Fit the Ridge regression model on the training data.
# This model has a parameter alpha. Let's try the following values: [0, 0.01, 0.1, 1, 10]
# Which of these alphas leads to the best RMSE on the validation set? Round your RMSE scores to 3 decimal digits.
df_full_train, df_test = train_test_split(df_extracted, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
len(df_train), len(df_val), len(df_test)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

del df_train['price']
del df_val['price']
del df_test['price']

dv = DictVectorizer(sparse=False)

categorical = list(df_train.dtypes[df_train.dtypes == 'object'].index)
numerical = list(df_train.dtypes[df_train.dtypes != 'object'].index)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

for alpha in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    yhat = model.predict(X_val)
    # rms = mean_squared_error(y_val, yhat, squared=False)
    rms = rmse(y_val, yhat)
    print(f"Alpha value of {alpha} resulted in RMSE of {rms}")

# alpha value of 0.01
