# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:08:25 2024

@author: H P
"""

# K-MEAN CLUSTERING ASSIGNMENTS

################ ASSIGNMENT NO : 1 ###############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a = pd.read_csv(r"C:\Users\H P\OneDrive\Documents\clustering\AutoInsurance.csv")
a
a.isna().sum()
a.drop(["Customer"],axis = 1, inplace = True)


numerical_columns = a.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    plt.boxplot(a[column])
    plt.title("Boxplot of "+column)
    plt.show()
    
    IQR = a[column].quantile(0.75) - a[column].quantile(0.25)
    lower_limit = a[column].quantile(0.25) - (1.5 * IQR)
    upper_limit = a[column].quantile(0.75) + (1.5 * IQR)
    a[column] = np.clip(a[column], lower_limit, upper_limit)
    
    plt.boxplot(a[column])   
    plt.title("New plot of "+column)
    plt.show()
    
a.isna().sum()
a.drop(["Number_of_Open_Complaints"],axis = 1, inplace = True)
#DUMMIES
data_new = pd.get_dummies(a, drop_first=True)
new_data = data_new.astype(int)
#NORMALIZATION
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return(x)
norm_data = norm_func(new_data)
#KMean Clustering
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_data)
    TWSS.append(kmeans.inertia_)
TWSS
# Scree plot
plt.plot(k, TWSS, marker='o', linestyle='--');plt.xlabel("No_of_Clusters");plt.ylabel("total_withn_SS")
# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(norm_data)
model.labels_
mb = pd.Series(model.labels_)
new_data["Clust"] = mb
#Columns Order changing
data = new_data.iloc[:,[22,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
#Aggregate means of each Clusters
new_data.iloc[:,1:].groupby(new_data.Clust).mean()

########### ASSIGNMENT NO : 2 ################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
b = pd.read_csv(r"C:\Users\H P\OneDrive\Documents\clustering\crime_data.csv")
b
b.place.unique()
b.columns
b.drop(["place"],axis = 1, inplace = True)
b.isna().sum()
#NORMALIZATION
def norm_func(i):
    x =(i-i.min()) / (i.max()-i.min())
    return(x)
norm_data = norm_func(b)
#Kmean Clustering
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_data)
    TWSS.append(kmeans.inertia_)
TWSS
#Scree plot
plt.plot(k, TWSS, 'g^-');plt.title('Scree plot of Kmean');plt.xlabel('no of clusters');plt.ylabel('Total within SS')
#Slecting 4 clusters
model = KMeans(n_clusters = 4)
model.fit(norm_data)
model.labels_
mb = pd.Series(model.labels_)
b['Clust'] = mb
#Columns Order Changing
data = b.iloc[:,[4,0,1,2,3]]

################## ASSIGNMENT NO : 3 #####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
c = pd.read_excel(r"C:\Users\H P\OneDrive\Documents\clustering\Telco_customer_churn.xlsx")
c
c.columns
c.Quarter.unique()
c.drop(['Customer_ID', 'Count','Quarter', 'Total_Refunds'],axis = 1, inplace = True)
c.isna().sum()
#MISSING VALUE IMPUTE
from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
c['Offer'] = pd.DataFrame(mode_imputer.fit_transform(c[['Offer']]))
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
c['Internet_Type'] = pd.DataFrame(mode_imputer.fit_transform(c[['Internet_Type']]))
#DUMMIES
data = pd.get_dummies(c, drop_first=True)
new_data = data.astype(int)
#NORMALIZATION
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return(x)
norm_data = norm_func(new_data)
#Kmeans Clustering
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_data)
    TWSS.append(kmeans.inertia_)
TWSS
#Scree plot o elbow curve
plt.plot(k, TWSS, 'ro--');plt.title('Scree plot of Kmean');plt.xlabel('no of clusters');plt.ylabel('Total within SS')
#Selecting 3 Clusters from the above scree plot
model =KMeans(n_clusters = 3)
model.fit(norm_data)
model.labels_
mb = pd.Series(model.labels_)
new_data['Cust'] = mb

################ ASSIGNMENT NO : 4 ################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
d =pd.read_excel(r"C:\Users\H P\OneDrive\Desktop\EastWestAirlines.xlsx")
d
d.columns
d.cc2_miles.unique()
d.cc1_miles.unique()
d.cc3_miles.unique()
d.drop(['ID#'],axis = 1, inplace = True)
#NORMALIZATION
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return(x)
norm_data = norm_func(d)
#Kmean Clustering
TWSS = []
k= list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters =i)
    kmeans.fit(norm_data)
    TWSS.append(kmeans.inertia_)
TWSS
#Scree plot or elbow curve
plt.plot(k, TWSS, 'b*:');plt.title('Scree plot of Kmean');plt.xlabel('no of Clusters');plt.ylabel('Total within SS')
#Selecting  4 Clusters from the above Scree plot
model = KMeans(n_clusters = 4)
model.fit(norm_data)
model.labels_
mb = pd.Series(model.labels_)
d['Clust'] = mb
