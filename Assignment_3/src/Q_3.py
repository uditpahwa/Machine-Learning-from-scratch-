import random
import pandas as pd
import numpy as np
#calculating the cos distance defined in the assignment
def cos_dis(x,y):
    temp = np.dot(np.transpose(x),y)/((np.linalg.norm(x))*(np.linalg.norm(y)))
    temp = np.exp(-temp)
    return(temp)
#this code gives the kmeans clusters
def k_means(centroid,X,k):
    x = 1000 #arbitrary 
    while x > 0.00001: # sum of the absolute error of the centroid is less than a threshold
        clusters=[]
        cluster_indices=[]
        for i in range(k):
            clusters.append([])
            cluster_indices.append([])
        for i in range(np.shape(X)[0]):
            dist=[]
            for j in range(len(centroid)):#calculating the distance of every point in the dataset from the centroids
                dist.append(cos_dis(X[i],centroid[j]))
            identity = np.argmin(np.array(dist))# identifying the minimum distance
            clusters[identity].append((list(X[i]))) # assigning that element to that point's cluster
            cluster_indices[identity].append(i)# assigning the index of that element to that point's cluster
        new_centroids =[]
        for i in range(k):#calculating new centroid
            new_centroids.append(np.average(np.array(clusters[i]),axis=0))
        new_centroids = np.array(new_centroids)
        x = np.sum(abs(new_centroids[1] - centroid[1]))
        centroid = new_centroids
    print('K_Means clustering done')
    return(cluster_indices)
data=pd.read_csv('tfidf_data.csv')
data.dropna(inplace=True)
X = data.drop(['Unnamed: 0','label'],axis=1)
X.dropna(inplace=True)
X= X.values
k=8
list_ = random.sample(range(1, np.shape(X)[0]),k)# randomly genrating index
centroid=[]
for i in range(len(list_)):# random centroids
    centroid.append(X[list_[i]])
centroid = np.array(centroid)

clu = k_means(centroid,X,k)

clu.sort(key= lambda x: x[0])#sorting the clusters according to the minimum index present in each cluster.
f= open('kmeans.txt',mode='w')
for i in range(8):
    f.write(str(clu[i]))
    f.write('\n')
f.close()