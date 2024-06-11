import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)  # Print the max columns from the dataframe

DF = pd.read_csv('banking.csv')     # read the csv / create a dataframe object
print(DF.shape) # print how many rows and cols the DF has

# Check if there are any missing values
print(DF.isnull().any(), '\n')    # isnull.any() checks each column; 'False' means no missing values
# print(DF.isnull().sum())  # option 2; 0 means no missing values

# Create a count plot showing how many people accepted/declined the offer
plt.figure()    # create a figure
sn.countplot(data=DF, x='y', hue='y')   # create countplot; hue adds color to 'y' col

# Create a countplot showing how many people from each job type accepted/declined the offer
plt.figure()    # create a figure
sn.countplot(data=DF, y='job', hue='job')   # create countplot

# Create a model / Initialize features/target
X_nonDummy = DF[['job', 'marital', 'default', 'housing', 'loan', 'poutcome']]  # features
y = DF[['y']]   # target
# X_nonDummy = DF[DF.columns[[1, 2, 4, 5, 6, 14]]]   # another option to get dummies

# Get dummy variables from X
X = pd.get_dummies(data=X_nonDummy, drop_first=True)

# Check if X transformed properly
print('First X_nonDummy: \n', X_nonDummy.head(1), '\n') # print first non-dummy data point
print('First X_dummy: \n', X.head(1), '\n') # print first dummy data point

from sklearn.model_selection import train_test_split    # import train_test_split class to train and test model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024, stratify=y) # set test size to 30%

from sklearn.linear_model import LogisticRegression # import LogisticRegression class to create model
model = LogisticRegression()    # instantialize model
model.fit(X_train, y_train)     # fit the model

y_pred = model.predict(X_test)  # predict y from test set

from sklearn import metrics # import metrics class for confusiom matrix and accuracy score
# metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)         # option 2
metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)    # passes X_test and y_test through the model

# How accurate is your model?
print(model.score(X_test, y_test))                    # option 1
print(metrics.accuracy_score(y_test, y_pred), '\n')   # option 2

# What are the precision, recall, f1-score, and support values?
print(metrics.classification_report(y_test, y_pred), '\n')  # print classification report for requested stats

print(model.predict(X_test.head(5)))    # print the predictions of the model on the first five rows of the test dataset
print(model.predict_proba(X_test.head(5)))  # print the probability estimates for each class based on model's predictions
# item 5 is 'yes' because probability is greater than 50%

plt.show()  # display all plots
