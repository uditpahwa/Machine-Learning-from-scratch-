import pandas as pd
import numpy as np
from numpy import log2 as log
data = pd.read_csv('ID3_dataset.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
#calculating entropy
def entropy_node(data):
    determine = data.keys()[-1]
    l = data[determine].unique()
    a = data[determine].value_counts()
    entropy = 0
    for i in l:
        prob = a[i]/len(data[determine])
        entropy = entropy-prob*log(prob)
    return(entropy)
def attribute_entropy(data,attribute):
    determine = data.keys()[-1]
    l = data[attribute].unique()
    attribute_entropy = 0
    for i in l:
        entropy_atr_class = entropy_node(data[data[attribute]==i])
        attribute_entropy += entropy_atr_class*len(data[data[attribute]==i][determine])/len(data[determine])
    return(attribute_entropy)

## finding the maximum gain = entropy_node - attribute entropy
def max_gain_class(data):
    determine = data.keys()[-1]
    attributes = data.keys()[:-1]
    info_gain = {}
    for i in attributes:
        temp = entropy_node(data) - attribute_entropy(data,i)
        info_gain[i] = temp
    winner_class = max(info_gain,key=info_gain.get)
    return(winner_class)

def decision_tree(data,tree=None):
    determine = data.keys()[-1]

    node = max_gain_class(data)
    #the components of the max gain class
    l = data[node].unique()
    if tree is None:
        tree = {}
        tree[node] ={}
    for i in l:
        sub_data = data[data[node]==i].reset_index(drop=True)
        y = len(sub_data.columns)
        if len(sub_data[determine])<=10 or y==2:#here y==2 stops the tree when only one feature is remaining in the feature set, as there will be no features in future iterations
            tree[node][i] = sub_data[determine].mode()[0]# assigns the mode of the remaining data to the leaf
        else:
            tree[node][i]=decision_tree(sub_data.drop(node,axis=1))#here we drop the current node feature from the sub_data and pass it for further split
    return(tree)
def predict(data,tree):#this is a code to traverse the tree dictionary
    temp = tree[list(tree.keys())[0]][data[list(tree.keys())[0]]]
    if (temp ==1) or (temp ==0) or (temp ==2):
        return(temp)
    else:
        tree =tree[list(tree.keys())[0]][data[list(tree.keys())[0]]]
        temp=predict(data,tree)
        return(temp)
# getting the tree
tree =decision_tree(data)
#getting the predictions
pred=[]
for i in range(len(data)):
    pred.append(predict(data.iloc[i],tree))
#Checking the performance of ID3 algorithm
print('The metrics for ID3 algorithm are::')
predictions = pd.DataFrame(pred,columns=['predictions'])
predictions['true'] = data['quality']
from sklearn.metrics import classification_report
print(classification_report(predictions['true'],predictions['predictions']))
#Computing the Decision Tree Classifier Using Scikitlearn
print('The metrics for scikit learn implementation with a min_samples_split =10:: ')
from sklearn.tree import DecisionTreeClassifier
dr = DecisionTreeClassifier(min_samples_split=10)
dr.fit(data.drop('quality',axis=1),data['quality'])
pred_=dr.predict(data.drop('quality',axis=1))
print(classification_report(predictions['true'],pred_))
from sklearn.model_selection import cross_val_score
l = (data.drop('quality',axis=1)).columns
precision_=cross_val_score(dr,data[l],data['quality'],cv=3,scoring = 'precision_macro')
recall_=cross_val_score(dr,data[l],data['quality'],cv=3,scoring = 'recall_macro')
accuracy_=cross_val_score(dr,data[l],data['quality'],cv=3,scoring = 'accuracy')
print('The metrics for a 3 fold cross validation on Decision tree classifier model')
print('\n')
print('Macro precision:',sum(precision_)/3)
print('Macro recall:',sum(recall_)/3)
print('accuracy:',sum(accuracy_)/3)

