# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import Libraries Import necessary libraries: pandas for data handling and matplotlib.pyplot for visualization.
Step 2: Load the Dataset Read the dataset from the provided CSV file.
Step 3: Explore the Dataset Display the first few records using head().
Show information about data types and missing values using info().
Check for null values
Step 4: Determine Optimal Number of Clusters (Elbow Method) Use the K-Means algorithm with different values of k (from 1 to 10).
Compute and store Within-Cluster-Sum-of-Squares (WCSS) for each k.
Plot WCSS vs number of clusters to find the "elbow point".
Step 5: Apply K-Means Clustering with Optimal k=5 Fit the KMeans model with 5 clusters.
Predict the cluster for each data point.
Add the predicted cluster labels to the original dataset.
Step 6: Separate the Data into Clusters Filter data points for each cluster into separate DataFrames.
Step 7: Visualize the Clusters Plot the clusters using a scatter plot with different colors for each cluster.
Use "Annual Income (k$)" and "Spending Score (1-100)" as axes.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: TAMIZHSELVAN B
RegisterNumber:  212223230225
*/
```

```
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")
data

data.head()

data.info

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
plt.show()

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)
y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="teal",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="green",label="cluster4")
plt.legend()
plt.title("Customer Segments")
print("Name:TAMIZHSELVAN B")
print("Reg NO: 212223230225")

```



## Output:
### PREVIEW DATASETS

![image](https://github.com/user-attachments/assets/a858a043-fe23-45c0-bdf7-4b01a493dddc)


### DATA.HEAD():

![image](https://github.com/user-attachments/assets/bb3daeb0-5867-43f5-9024-d18af1718c2e)


### DATA.INF0():

![image](https://github.com/user-attachments/assets/50e8811c-0895-4a48-80ee-56681a4c9504)


### DATA.ISNULL().SUM():


![image](https://github.com/user-attachments/assets/755bf5e0-c77a-4d9c-9ed2-92f427ff7bff)


### PLOT USING ELBOW METHOD:

![image](https://github.com/user-attachments/assets/d3186b5e-df4b-46b7-bf4d-18d0a41e8367)


### Y_PRED ARRAY:

![image](https://github.com/user-attachments/assets/5f35af58-d896-40eb-8bc9-31fb402f795f)


### CUSTOMER SEGMENT:


![image](https://github.com/user-attachments/assets/516b0e0f-b981-4a1b-be78-b4e4e83028c0)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
