CS7CS4 - Linear Regression Compettion Entry
===========================================

**Entry for TCD Machine Learning Kaggle Competition**

This is a python script which performs a linear regression to predict incomes based on a variety of features. This approach makes use of pipelines to streamline the preprocessing and regression steps in a convenient manner. Verbose output is given through the pipelines allowing for status updates on progress

**Usage**

Provided sklearn-python and numpy-python are present, this script should run on any machine provided data is contained in a folder. Progress through the script is displayed on the terminal, followed by RMSE.

**Issues**

This script does allow for prediction of income, with a RMSE of close to 80,000 according to the leaderboards on Kaggle. Improvements could be made by better cleaning the data, possibly some more exploratory analysis on where there is correlation between features. 

Categorical data also posed somewhat of an issue, although onehot encoding seems like a suitable solution, the simple fact that there are some labels in the prediction features that are not present in the training data is a major source of error. Investigation into more suitable encoding methods may be necessary.

**TODO (or Future Work)**
    
    * Clean Data Further
    * Further Investigation on Models
    * Implement CV
