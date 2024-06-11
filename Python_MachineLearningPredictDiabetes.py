import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

# 1. Create a DataFrame “diabetes_knn” to store the diabetes data
diabetes_knn = pd.read_csv('diabetes_data.csv')
# and set option to display all columns without any restrictions on the number of columns displayed.
pd.set_option('display.max_columns', None)


# 2. Determine the dimensions of the “diabetes_knn” dataframe.
print('Dimensions of "diabetes_knn" dataframe:', diabetes_knn.shape)

# 3. Create the Feature Matrix and Target Vector.
X = diabetes_knn.iloc[:, :-1] # features / all rows, all columns except last col 'Outcome'
y = diabetes_knn.iloc[:, -1]  # target / all rows, only last column 'Outcome'

# 4. Standardize the attributes of Feature Matrix
from sklearn.preprocessing import StandardScaler    # import StandardScaler class
scaler = StandardScaler()                           # Instantiate the scaler
scaler.fit(X)                                       # Scale the features
diabetes_norm = pd.DataFrame(data=scaler.transform(X), columns=X.columns)   # Create dataframe for scaled data

# 5. Split the Feature Matrix and Target Vector into train A (70%) and train B sets (30%).
# Use random_state=2024, and stratify based on Target vector.
from sklearn.model_selection import train_test_split    # import train_test_split class
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(diabetes_norm, y, random_state=2024, stratify=y, test_size=0.3)    # Split the train set to A and B

# 6. Develop a KNN based model and obtain KNN score (accuracy) for train A and train B data for k’s values ranging between 1 to 9.
from sklearn.neighbors import KNeighborsClassifier  # import KNeighborsClassifiers class
from sklearn import metrics # import metrics class

neighbors = np.arange(1,10) # create a range of neighbors from 1-9
trainA_Accuracy = []    # create empty list for train A accuracy scores
trainB_Accuracy = []    # create empty list for train B accuracy scores

# Create for loop to iterate through different k values
for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)  # instantiate the model
    model.fit(X_trainA, y_trainA)       # fit the model to train set A
    y_predA = model.predict(X_trainA)   # predict y from train A
    y_predB = model.predict(X_trainB)   # predict y from train B
    trainA_Accuracy.append(metrics.accuracy_score(y_trainA, y_predA)) # add accuracy to train A list
    trainB_Accuracy.append(metrics.accuracy_score(y_trainB, y_predB)) # add accuracy to train B list

# 7. Plot a graph of train A and train B score and determine the best value of k.
plt.figure()    # create a figure
plt.plot(neighbors, trainA_Accuracy, label='Train A') # plot k vs Accuracy score
plt.plot(neighbors, trainB_Accuracy, label='Train B') # plot k vs Accuracy score
plt.xlabel('Number of Neighbors (k)')  # label  x-axis
plt.ylabel('Accuracy')  # label  y-axis
plt.xticks(neighbors)   # display x-ticks
plt.title('K vs. Accuracy for Train A and B Datasets')  # plot title
plt.legend()    # display legend

# Based on the results, k=8 is the best value for k since the model achieves the highest accuracy for the Train B set at k=8.

# 8. Display the score of the model with best value of k.
newModel = KNeighborsClassifier(n_neighbors=8)  # new model with new k value
newModel.fit(X_trainA, y_trainA)      # fit the model to train A set
y_pred = newModel.predict(X_trainB)   # predict if the patient has diabetes with X_trainB set
print('Accuracy of Model with k=8:', newModel.score(X_trainB, y_trainB)) # print accuracy of model w/ new k

# Also print and plot the confusion matrix for Train B, using Train A set as the reference set for training.
# Plot confusion matrix:
metrics.ConfusionMatrixDisplay.from_predictions(y_trainB, y_pred)   # create confusion matrix comparing actual (y_trainB) vs predicted (y_pred)
# Print confusion matrix:
conf_matrix = metrics.confusion_matrix(y_trainB, y_pred)    # instantiate confusion matrix for printing
print('Confusion matrix results:\n', conf_matrix)   # print confusion matrix

# 9. Predict the outcome for a person with the given attributes.
sample = pd.DataFrame(data=np.array([[1, 150, 60, 12, 300, 28, 0.4, 45]]), columns=X.columns)   # create sample
sampleScaled = pd.DataFrame(data=scaler.transform(sample), columns=X.columns)   # scale the sample
print('Prediction: ', newModel.predict(sampleScaled))

# Since the program outputted '1', this person is predicted to have diabetes.

print('\nClassification report:\n', metrics.classification_report(y_trainB, y_pred))


plt.show()  # display all plots

