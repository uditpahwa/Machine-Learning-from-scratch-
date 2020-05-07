import pandas as pd
import numpy as np

data = pd.read_csv('winequality-red.csv')

data['quality'] = 1*data['quality'].apply(lambda x: True if(x>6) else False,)

#min max scaling
#X = X-Xmin/Xmax-Xmin
l = (data.drop('quality',axis=1)).columns
for i in l:
    data[i] = data[i].apply(lambda x: (x-min(data[i]))/(max(data[i])-min(data[i])))
data.to_csv('Logistic_regression.csv')
# MAKING THE DATA SET FOR ID3 ALGORITHM
data_ = pd.read_csv('winequality-red.csv')
for i in range(len(data_['quality'])):
    if data_['quality'].iloc[i] < 5:
        data_['quality'].iloc[i]=0
    elif ((data_['quality'].iloc[i] == 5) or (data_['quality'].iloc[i] == 6)):
        data_['quality'].iloc[i] = 1
    else:
        data_['quality'].iloc[i]=2
#NORMALIZING THE DATA 
for i in l:
    data_[i]=data_[i].apply(lambda x: (x-data_[i].mean())/np.std(data_[i]))
#DIVIDING DATA INTO BINS 0,1,2,3
for i in l:
    bin_size = (max(data_[i]) - min(data_[i]))/4
    start = min(data_[i])
    b_1 = start + bin_size
    b_2 = b_1+bin_size
    b_3 = b_2+bin_size
    end = max(data_[i])
    for j in range(len(data_[i])):
        if data_[i].iloc[j]>=start and data_[i].iloc[j]<b_1:
            data_[i].iloc[j] = 0
        elif data_[i].iloc[j]>=b_1 and data_[i].iloc[j]<b_2:
            data_[i].iloc[j] = 1
        elif data_[i].iloc[j]>=b_2 and data_[i].iloc[j]<b_3:
            data_[i].iloc[j] = 2
        else:
            data_[i].iloc[j] = 3
data_.to_csv('ID3_dataset.csv')
