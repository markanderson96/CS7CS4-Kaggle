import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

dataset = pd.read_csv('data/tcd ml 2019-20 income prediction training (with labels).csv')
features = ['Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses', 'Hair Color', 'Body Height [cm]']

x_data = dataset[features]
x_data[:,1] = LabelEncoder().fit_transform(x_data[:,1])
print(x_data)
y_data = dataset['Income in EUR']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor.intercept_)
print(regressor.coef_)
