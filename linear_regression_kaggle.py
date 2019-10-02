import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics

dataset = pd.read_csv('data/tcd ml 2019-20 income prediction training (with labels).csv')

numerical_features = ['Year of Record','Age','Size of City','Wears Glasses','Body Height [cm]']
categorical_features = ['Gender','Country','Profession','University Degree','Hair Color']

numerical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler', StandardScaler())], 
    verbose=True)
categorical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('Onehot', OneHotEncoder(handle_unknown='ignore'))],
    verbose=True)

preprocessor = ColumnTransformer(
        transformers=[
            ('Numerical Data', numerical_transformer, numerical_features),
            ('Categorical Data', categorical_transformer, categorical_features)],
        verbose=True)

lr = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('Linear Regression', LinearRegression())],verbose=True)

x_data = dataset.drop(['Instance','Income in EUR'], axis=1)
y_data = dataset['Income in EUR']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

lr.fit(x_train, y_train)

print(lr.score)
