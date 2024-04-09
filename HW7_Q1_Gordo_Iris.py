# Iris Gordo
# ITP 449
# HW7
# Q1

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)  # display max columns when printing dataframes

DF = pd.read_csv('insurance.csv')   # 1. Read csf file / save to dataframe

# 2. Create a category plot to study the impact of sex and smoking on medical costs.
sn.catplot(data=DF, x='sex', y='charges', col='smoker', hue='sex', sharey=True, kind='bar', errorbar=None)   # create a category plot

# 3. Create scatterplots to study the impact of age and bmi on medical costs.
plt.figure()    # create a figure for the following plot
plt.scatter(data=DF, x='age', y='charges')    # create a scatter plot of age vs charges
plt.xlabel('age')       # label x-axis
plt.ylabel('charges')   # label y-axis

plt.figure()    # create a figure for the following plot
plt.scatter(data=DF, x='bmi', y='charges')    # create a scatter plot of bmi vs charges
plt.xlabel('bmi')       # label x-axis
plt.ylabel('charges')   # label y-axis

# Build a linear model that takes 'age', 'bmi', and 'smoker' as inputs and predicts the 'charges'
# 4. Define the feature and target dataframes
X = DF[['age', 'bmi', 'smoker']]    # features
y = DF['charges']   # target

# 5. Since 'smoker' is a categorical feature, convert it into a dummy feature.
X_dummy = pd.get_dummies(data=X, columns=['smoker'], drop_first=True)

# 6. Partition the data into train and test sets (70/30)
from sklearn.model_selection import train_test_split    # import train_test_split class
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.3, random_state=2024)   # split the data into train and test sets

from sklearn.linear_model import LinearRegression   # import LinearRegression class
model = LinearRegression()  # create a Linear Regression model

# 7. Fit the model to a linear regression model
model.fit(X_train, y_train)

# 8. Predict the medical cost in the test set.
y_pred = model.predict(X_test)  # Predict the medical cost in the test set

# Create a scatter plot between the actual and predicted medical costs
plt.figure()    # create a figure
plt.scatter(x=y_test, y=y_pred) # create scatter plot
plt.xlabel('Actual Cost')   # label x-axis
plt.ylabel('Predicted Cost')    # label y-axis

# 9. Calculate the score of the model for the test set.
print('Model Score:', model.score(X_test, y_test), '\n')

# get the accuracy score (not part of HW, just curious)
from sklearn import metrics
print('Model Accuracy score: ', metrics.accuracy_score(y_test, y_pred))

# 10. Predict the medical cost of a 51 year old non-smoker with bmi=29.1.
print(X_dummy.columns, '\n')    # Print the column labels of X_dummy

# Create 2D numpy array that has the values in the corresponding order
sampleData = np.array([[51, False, 29.1]])

# Make dataframe whose values are coming from the array and its column labels match X_dummy columns
sample = pd.DataFrame(data=sampleData, columns=X_dummy.columns)

# Predict the medical cost given the inputs
print('Predicted medical cost of 51 y/o non-smoker with BMI 29.1: ', model.predict(sample))

# What if the same person was a smoker? (Follow the same steps but make 'smoker_yes' = True)
sampleData2 = np.array([[51, True, 29.1]])  # create array with desired inputs
sample2 = pd.DataFrame(data=sampleData2, columns=X_dummy.columns)   # convert array to dataframe
print('Predicted medical cost of 51 y/o smoker with BMI 29.1: ', model.predict(sample2))    # print the prediction based on inputs

plt.show()  # display all plots