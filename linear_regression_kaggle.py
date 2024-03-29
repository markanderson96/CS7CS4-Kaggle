import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from math import sqrt

# Create pandas dataframes for training data and prediction data
dataset_training = pd.read_csv('data/tcd ml 2019-20 income prediction training (with labels).csv')
dataset_predict = pd.read_csv('data/tcd ml 2019-20 income prediction test (without labels).csv')

# Labels of numerical and categorical features
numerical_features = ['Year of Record','Age','Size of City','Wears Glasses','Body Height [cm]']
categorical_features = ['Gender','Country','Profession','University Degree','Hair Color']

# Transform the respective features
# Impute and Scale numerical features
numerical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median', verbose=1)),
    ('Scaler', StandardScaler())], 
     verbose=True)
# Impute and One Hot Encode categorical features
categorical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='constant', fill_value='missing',verbose=1)),
    ('Onehot', OneHotEncoder(handle_unknown='ignore',sparse=True))],
    verbose=True)
# Preprocessor operations
preprocessor = ColumnTransformer(
        transformers=[
            ('Numerical Data', numerical_transformer, numerical_features),
            ('Categorical Data', categorical_transformer, categorical_features)],
        verbose=True)

# Linear Regression Pipeline: Preprocess -> Ridge Regression
lr = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('Ridge Regression', Ridge(alpha=0.5,fit_intercept=True,solver='sag'))],
    verbose=True)

# Create x features and y features
x_data = dataset_training.drop(['Instance','Income in EUR'], axis=1)
y_data = dataset_training['Income in EUR']

# Split data 70/30
x_train, x_test, y_train, y_real = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
# Drop any unnecessary data from prediction data
x_predict = dataset_predict.drop(['Instance', 'Income'], axis=1)

model = lr.fit(x_train, y_train)
y_test = lr.predict(x_test)

# Output predictions to csv file
# Currently this must be manually copied over to CSV file containing instances but this could be automated
predictions = pd.DataFrame(data=lr.predict(x_predict))
predictions.to_csv('data/pred.csv')

#Print RMSE
MSE_test = mean_squared_error(y_test, y_real)
RMSE_test = sqrt(MSE_test)
print('RMSE: ', RMSE_test)
