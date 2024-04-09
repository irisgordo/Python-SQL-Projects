import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)  # display max columns when printing dataframes

# 1. Read csv file / save to dataframe / import header
DF = pd.read_csv('wineQualityReds.csv', header=0)

# 2. Drop 'Wine' from the dataframe.
DF.drop(columns='Wine', inplace=True)

# 3. Extract 'quality' and store it in a separate variable.
quality = DF['quality']

# 4. Drop 'quality' from the dataframe
DF.drop(columns='quality', inplace=True)

# 5. Print the dataframe and 'quality'.
print('Updated DF:', '\n', DF, '\n')
print('Quality:', '\n', quality, '\n')

# 6. Normalize all columns of the dataframe. Use the MinMaxScaler class from sklearn.preprocessing.
from sklearn.preprocessing import MinMaxScaler  # import MinMaxScaler class
scaler = MinMaxScaler() # create a scaler object
scaler.fit(DF)  # fit the data to the scaler

# 7. Print the normalized dataframe.
DFNorm = pd.DataFrame(scaler.transform(DF), columns=DF.columns) # create a dataframe of the normalized data
print('Normalized dataframe:', '\n', DFNorm, '\n')

# 8. Create a range of k values from 1-20 for k-means clustering.
# Iterate on the k values and store the inertia for each clustering in a list.
# Pass random_state=2024 and n_init='auto' to KMeans()
from sklearn.cluster import KMeans  # import KMeans class

kRange = np.arange(1,21)    # create a range of k values from 1-20
inertiaVal = [] # create empty list for model's inertia to be stored

# Create for loop to iterate through multiple clusters
for k in kRange:
    model = KMeans(n_clusters=k, random_state=2024, n_init='auto')    # create the KMeans model w/ given parameters
    model.fit(DFNorm)   # fit the model with the normalized data
    inertiaVal.append(model.inertia_)   # add the model's inertia at each cluster to the list

# 9. Plot the chart of Inertia vs Number of Clusters.
plt.figure()    # create a figure
plt.plot(kRange, inertiaVal, marker='o')  # plot x=cluster number k, y=inertia
plt.xlabel('Number of Clusters, k') # label x-axis
plt.ylabel('Inertia')   # label y-axis
plt.xticks(np.arange(1,21, step=1)) # make x-ticks in increments of 1 within range

'''
10. What K (number of clusters) would you pick for kmeans?
        I would choose 6 clusters since there are 6 different quality levels and it
        would be efficient to cluster different wines based on their quality for analysis.
        Moreover, at k=6, the decrease in inertia begins to slow down. This is an indicator
        that k=6 is optimal since it has low inertia and is a low number for clusters.
'''

# 11. Now cluster the wines into K=6 clusters. Use random_state = 2024 and n_init='auto' when you
# instantiate the k-means model. Assign the respective cluster number to each wine. Print the
# dataframe showing the cluster number for each wine.
newModel = KMeans(n_clusters=6, random_state=2024, n_init='auto')   # instantiate a new k-means model
newModel.fit(DFNorm)    # fit the model to the normalized data

DFNorm['Cluster'] = newModel.labels_   # create new col 'clusters' to pair each wine to a cluster number
print('DFNorm with clusters column added:', '\n', DFNorm, '\n')  # print DFNorm

# 12. Add the quality back to the dataframe. (1)
DFNorm['Quality'] = quality # create col 'Quality' / assign to quality variable saved earlier
print('DFNorm with quality column added:', '\n', DFNorm, '\n')    # print DFNorm

# 13. Now print a crosstab (from Pandas) of cluster number vs quality.
# Comment if the clusters represent the quality of wine. (3)
print('Crosstab of Cluster Number vs Quality:')
print(pd.crosstab(DFNorm['Quality'], DFNorm['Cluster']))   # print a crosstab of cluster number vs quality
'''
I don't think that the clusters represent the quality of wine well since qualities 5 and 6 seem to dominate each cluster.
There is little to no variation among each cluster. 
Ideally, it was expected that each quality would each have one cluster, however, each quality existed in each cluster.
'''


plt.show()  # display all plots