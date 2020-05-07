import pandas as pd
import numpy as np
data = pd.read_csv('AllBooks_baseline_DTM_Labelled.csv')
data.drop(13,axis=0,inplace=True)
data = data.reset_index(drop=True) #resetting the index due to removal of one item line
data['Unnamed: 0']=data['Unnamed: 0'].apply(lambda x: x.split('_')[0]) #remooving the _ch# line from the labels thus obtaining 8 labels
a=data.sum()
l = (data.drop('Unnamed: 0',axis=1)).columns # obtaining the list of rest of the column names
##creating the 
for i in l: # here we calculate the tf idf matrix as defined in the assignment
    data[i] = data[i].apply(lambda x: x*np.log((1+data.shape[0])/(1+a[i])))
#here we are converting the tf idf matrix into the standard form
X = data.drop('Unnamed: 0',axis=1)
x =X.values
#calculating the L2 norm for the vector
e = np.linalg.norm(x,axis=1)
#calculating standard vectors
for i in range(589):
    x[i] = x[i]/e[i]
data_ = pd.DataFrame(x,columns=l)
data_['label'] = data['Unnamed: 0']
data_.to_csv('tfidf_data.csv')