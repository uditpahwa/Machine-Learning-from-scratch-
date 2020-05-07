import pandas as pd
import numpy as np
#implementing logistic regression

thresh_hold = 0.0000001#Adjust this for faster processing !
def Gradient_Descent(X_train,y,theta):
	lin_est = np.dot(np.transpose(theta),X_train) ## computing here the h(x)
	t_1 = np.dot(np.transpose(y),np.log(sigmoid(lin_est))) ##computing y*log(h(x))
	t_2 = np.dot(np.transpose(1-y),np.log(1-sigmoid(lin_est)))##computing (1-y)*h(x)
	err = -(t_1+t_2) 
	cost= err/np.shape(X_train)[1] ##the error term
	print('Starting Cost theta initiated as one:',cost)
	old = cost
	diff = cost
	while(diff >= thresh_hold):
		grad = np.dot(X_train,(sigmoid(lin_est)-y))
		grad = grad/np.shape(X_train)[1]
		alpha = 0.09
		theta = theta - alpha*grad
		lin_est = np.dot(np.transpose(theta),X_train)
		t_1 = np.dot(np.transpose(y),np.log(sigmoid(lin_est)))
		t_2 = np.dot(np.transpose(1-y),np.log(1-sigmoid(lin_est)))
		err = -(t_1+t_2)
		cost= err/np.shape(X_train)[1]
#		print('Iter_Cost: ',cost)
		diff = abs(cost-old)
		old = cost
	return(theta,cost,lin_est)

def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return(sigmoid)

def binary_func(pred_og):
    for i in range(len(pred_og)):
        if pred_og[i]>0.5:
            pred_og[i]=1
        else:
            pred_og[i]=0
    return(pred_og)
data = pd.read_csv('Logistic_regression.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
print('Implementing Logistic Regression')
l = (data.drop('quality',axis=1)).columns
feature_list = l 
feature = (data[feature_list].values) 
theta = np.ones(len(feature_list)+1) #INITIATING THETA
a=np.ones(len(feature)) #ADDING X(0)
X_train = np.insert(feature,0,a,axis=1) #ADDING X(0)
X_train =np.transpose(X_train)
y = np.array(data['quality'])
theta,cost,lin_est = Gradient_Descent(X_train,y,theta)
pred_og = sigmoid(lin_est)
pred_og = binary_func(pred_og)
print('The metrics for logistic regression using my implementation of logistic regression::')
from sklearn.metrics import classification_report
print(classification_report(y,pred_og))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='saga',penalty='none')
lr.fit(data.drop('quality',axis=1),data['quality'])
from sklearn.metrics import classification_report

pred = lr.predict(data.drop('quality',axis=1))
print('The metrics for logistic regression using SAGA solver::')
print(classification_report(y,pred))


preds = pd.DataFrame(pred_og,columns=['predicted_by_me'])
preds['predicted_by_not_me'] = pred
error_rate = (preds['predicted_by_me'] != preds['predicted_by_not_me']).mean()

print('The prediction mismatch between my model and the SAGA solver model is ::',error_rate)
print('\n')
print('Implementing Cross-Validation')
print('\n')

##3 fold cross validation for my model begins::

slot_1 = data.iloc[0:533]
slot_2 = data.iloc[533:1066]
slot_3 = data.iloc[1066:1599]

precision=[]
recall=[]
accuracy=[]
for i in range(1,4):
    if i == 1:
        test_set = slot_1
        train_set = slot_2.append(slot_3)
        feature_list = l #USED LIST OF FEATURES GETS INCREASED EVERY ITERATION
        feature = (train_set[feature_list].values) #GETTING VALUES FROM DATA FRAME AS AN ARRAY
        theta = np.ones(len(feature_list)+1) #INITIATING THETA
        a=np.ones(len(feature)) #ADDING X(0)
        X_train = np.insert(feature,0,a,axis=1) #ADDING X(0)
        X_train =np.transpose(X_train)
        y = np.array(train_set['quality'])
        params,cost,lin_est = Gradient_Descent(X_train,y,theta)
        feature_test = (test_set[feature_list].values)
        theta = params #PARAMS ARE THE FINAL ITERATION PARAMETERS
        a=np.ones(len(feature_test)) #ADDING X(0)
        X_test = np.insert(feature_test,0,a,axis=1)
        X_test =np.transpose(X_test)
        y_test = np.array(test_set['quality'])
        test_err = np.dot(np.transpose(theta),X_test)
        temp = binary_func(sigmoid(test_err))
        from sklearn.metrics import classification_report 
        X =classification_report(y_test,temp,output_dict=True)
        precision.append(X['1']['precision'])
        recall.append(X['1']['recall'])
        accuracy.append(X['accuracy'])
    
        
            
    if i == 2:
        test_set = slot_2
        train_set = slot_1.append(slot_3)
        feature_list = l #USED LIST OF FEATURES GETS INCREASED EVERY ITERATION
        feature = (train_set[feature_list].values) #GETTING VALUES FROM DATA FRAME AS AN ARRAY
        theta = np.ones(len(feature_list)+1) #INITIATING THETA
        a=np.ones(len(feature)) #ADDING X(0)
        X_train = np.insert(feature,0,a,axis=1) #ADDING X(0)
        X_train =np.transpose(X_train)
        y = np.array(train_set['quality'])
        params,cost,lin_est = Gradient_Descent(X_train,y,theta)
        feature_test = (test_set[feature_list].values)
        theta = params #PARAMS ARE THE FINAL ITERATION PARAMETERS
        a=np.ones(len(feature_test)) #ADDING X(0)
        X_test = np.insert(feature_test,0,a,axis=1)
        X_test =np.transpose(X_test)
        y_test = np.array(test_set['quality'])
        test_err = np.dot(np.transpose(theta),X_test)
        temp = binary_func(sigmoid(test_err))
        from sklearn.metrics import classification_report 
        X =classification_report(y_test,temp,output_dict=True)
        precision.append(X['1']['precision'])
        recall.append(X['1']['recall'])
        accuracy.append(X['accuracy'])
    if i == 3:
        test_set = slot_3
        train_set = slot_1.append(slot_2)
        feature_list = l #USED LIST OF FEATURES GETS INCREASED EVERY ITERATION
        feature = (train_set[feature_list].values) #GETTING VALUES FROM DATA FRAME AS AN ARRAY
        theta = np.ones(len(feature_list)+1) #INITIATING THETA
        a=np.ones(len(feature)) #ADDING X(0)
        X_train = np.insert(feature,0,a,axis=1) #ADDING X(0)
        X_train =np.transpose(X_train)
        y = np.array(train_set['quality'])
        params,cost,lin_est = Gradient_Descent(X_train,y,theta)
        feature_test = (test_set[feature_list].values)
        theta = params #PARAMS ARE THE FINAL ITERATION PARAMETERS
        a=np.ones(len(feature_test)) #ADDING X(0)
        X_test = np.insert(feature_test,0,a,axis=1)
        X_test =np.transpose(X_test)
        y_test = np.array(test_set['quality'])
        test_err = np.dot(np.transpose(theta),X_test)
        temp = binary_func(sigmoid(test_err))
        from sklearn.metrics import classification_report 
        X =classification_report(y_test,temp,output_dict=True)
        precision.append(X['1']['precision'])
        recall.append(X['1']['recall'])
        accuracy.append(X['accuracy'])
print('\n')
print('The metrics for a 3 fold cross validation on my logistic regression model')
print('precision:',sum(precision)/3)
print('recall:',sum(recall)/3)
print('accuracy:',sum(accuracy)/3)
print('\n')
from sklearn.model_selection import cross_val_score
l = (data.drop('quality',axis=1)).columns
precision_saga=cross_val_score(lr,data[l],data['quality'],cv=3,scoring = 'precision')
recall_saga=cross_val_score(lr,data[l],data['quality'],cv=3,scoring = 'recall')
accuracy_saga=cross_val_score(lr,data[l],data['quality'],cv=3,scoring = 'accuracy')
print('The metrics for a 3 fold cross validation on SAGA solver model')
print('precision:',sum(precision_saga)/3)
print('recall:',sum(recall_saga)/3)
print('accuracy:',sum(accuracy_saga)/3)