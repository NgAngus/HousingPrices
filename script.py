# Major Library Imports
import numpy as np
import pandas as pd
import os

## ML Imports
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

# Data Retrieval
df = pd.read_csv('../input/train.csv')

# Training Data
# Drop columns that hinder the model's efficacy
drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Id', 'BsmtFinSF2']
X = df.drop(drop_cols + ['SalePrice'], axis=1).copy()
y = df.loc[:, ['SalePrice']].copy()

# Submission Data
test = pd.read_csv('../input/test.csv').drop(drop_cols, axis=1).copy()
test_ids = pd.read_csv('../input/test.csv')['Id']

# Creation of columns indicating what will be imputed
cols_with_missing = (col for col in X.columns if X[col].isnull().any())
for col in cols_with_missing:
    X[col + '_was_missing'] = X.loc[:, col].isnull()
    test[col + '_was_missing'] = test.loc[:, col].isnull()
assert X.shape[1] == test.shape[1], "Imputation Indicator # Columns Mismatch"

# One-hot encode training predictors and align test data to it
one_hot_encoded_train_predictors = pd.get_dummies(X)
one_hot_encoded_train_predictors, final_test = one_hot_encoded_train_predictors.align(pd.get_dummies(test), join='inner', axis=1)

# Split Dataset
train_X, valid_X, train_y, valid_y = train_test_split(one_hot_encoded_train_predictors, y, random_state=7, test_size=0.15)

# Model Training
xgb = XGBRegressor(n_estimators=10000, learning_rate=0.119899942, n_jobs=4, max_depth=3, min_child_weight=1, gamma=0.0088, subsample=0.7, colsample_bytree=0.8)
model = xgb.fit(train_X, train_y, eval_metric="rmse", early_stopping_rounds=233, eval_set=[(valid_X, valid_y)], verbose=False)

# Create Submission Using Model
predictions = model.predict(final_test)
my_submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
my_submission.to_csv('submission.csv', index=False)