# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:39:04 2024

@author: H P
"""

# IRIS DATA
#K mean Clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = pd.read_csv(r"C:\SUHAIL\python data\IRIS.csv")
iris

# boxplot of all inputs
columns = iris.select_dtypes(include=[np.number]).columns.tolist()
for column in columns:
    plt.boxplot(iris[column])
    plt.title("Boxplot of "+column)
    plt.show()
# outliers removing using winsorizer
iris.columns
from feature_engine.outliers import Winsorizer
winsorizer = Winsorizer(capping_method='iqr',
                        tail='both',
                        fold=1.5,
                        variables=["sepal_width"])
iris["sepal_width"]=winsorizer.fit_transform(iris[["sepal_width"]])

plt.boxplot(iris.sepal_width)
# Drop column
iris.drop(["species"],axis=1,inplace=True)
# Missing value 
iris.isna().sum()
# Normalization
def nom_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return(x)
iris = nom_func(iris)
# K-Mean Clustering
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(iris)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel(" Number of Clusters");plt.ylabel("Total with in sum of squars")
# Selecting 3 Clusters from the above Scree plot
model = KMeans(n_clusters=3)
model.fit(iris)
model.labels_
mb = pd.Series(model.labels_)
iris["Cluster"] = mb
#
iris = iris.iloc[:,[4,0,1,2,3]]

###################
# NAIVE BIASE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

iris = pd.read_csv(r"C:\SUHAIL\python data\IRIS.csv")
iris
iris.head()

X=iris.iloc[:, 0:3] # Excluding id column 
Y = np.array(iris["species"]) # Target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)
y_pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames= ['Predictions'])
###################


import matplotlib.pyplot as plt

# Assuming Y_test and pred are arrays or lists containing the actual and predicted values
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.show()
############

