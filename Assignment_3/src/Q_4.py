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

#this is the cluster generating function giving 8 clusters 
def cluster_gen(old_cluster,dis_m):
    new_cluster = old_cluster
    while(len(new_cluster)>8): # alter the '8' here to get any number of clusters
        min_index = np.unravel_index(np.argmin(dis_m, axis=None), dis_m.shape)#obtaining the minimum distance vectors
        new_cluster[min_index[0]] = (new_cluster[min_index[0]]+new_cluster[min_index[1]]) # updating the cluster
        new_cluster.remove(new_cluster[min_index[1]])##cluster remove 
        dis_m = update_matrix(dis_m,min_index) # updating the matrix
    print('Agglomerative clustering done')
    return(new_cluster)
#this matrix updates the distance matrix using single linkage criteria
def update_matrix(dis_m,min_index): 
    dis_m[min_index[0]]= np.minimum(dis_m[min_index[0]],dis_m[min_index[1]])# takes the minimum of the row between the 2 min_index clusters of elements combined in the prev iteration 
    dis_m[:,min_index[0]]= np.minimum(dis_m[:,min_index[0]],dis_m[:,min_index[1]])#takes the minimum of the column between the 2 min_index clusters of elements combined in the prev iteration
    dis_m[min_index[0]][min_index[0]]=1000000000 # making the diagonal element a large positive
    temp= np.delete(dis_m,(min_index[1]),axis=0)      # dropping one of the 2 cluster/element
    new_dist= np.delete(temp,(min_index[1]),axis=1)
    
    return(new_dist)     
#calculating the distance
def distance_matrix(X,cluster):
    dist = np.zeros((len(cluster),len(cluster)))
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if i != j:
                dist[i][j] = cos_dis(X[i],X[j])
            else:
                dist[i][j] = 10000000000
    return(dist)    

data=pd.read_csv('tfidf_data.csv')
X = data.drop(['Unnamed: 0','label'],axis=1)
X= X.values
#performing pca analysis to reduce the size of features to 100
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(X)
y = pca.transform(X)

k=8
list_ = random.sample(range(1, np.shape(y)[0]),k)
centroid=[]
for i in range(len(list_)):
    centroid.append(y[list_[i]])
centroid = np.array(centroid)
cluster=list()
for i in range(np.shape(y)[0]):
    cluster.append([i])
dis =distance_matrix(y,cluster)
c=cluster_gen(cluster,dis)
clu = k_means(centroid,y,k)
c.sort(key= lambda x: x[0])
clu.sort(key= lambda x: x[0])
f= open('kmeans_reduced.txt',mode='w')
for i in range(8):
    f.write(str(clu[i]))
    f.write('\n')
f.close()
f= open('agglomerative_reduced.txt',mode='w')
for i in range(8):
    f.write(str(c[i]))
    f.write('\n')
f.close()