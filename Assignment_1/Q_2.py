import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
thresh_hold = 0.000000001   
def Gradient_Descent(X_train,y,theta): 
	err = np.dot(np.transpose(theta),X_train) - y
	err_sq = err**2
	cost = np.sum(err_sq)
	cost = (1/2)*cost
	cost= cost/np.shape(X_train)[1]
	old = cost
	diff = cost
	while(diff >= thresh_hold):
		grad = np.dot(X_train,err)
		grad = grad/np.shape(X_train)[1]
		alpha = 0.05
		theta = theta - alpha*grad
		pred = np.dot(np.transpose(theta),X_train)
		err =  pred - y
		err_sq = err**2
		cost = np.sum(err_sq)
		cost = (1/2)*cost
		cost= cost/np.shape(X_train)[1]
#		print('Iter_Cost: ',cost)
		diff = abs(cost-old)
		old = cost
	return(theta,cost,pred)
# importing the data
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
train_set.columns = ['feature','labels']
test_set.columns = ['feature','labels']
## COMPUTING THE FEATURES
for i in range(8):
	train_set['x'+str(i+2)] = train_set['feature'].apply(lambda x: x**(i+2))
	test_set['x'+str(i+2)] = test_set['feature'].apply(lambda x: x**(i+2))

# PART a
feature_list = []
l=[]
train_err=[]
test_err_=[]
l = train_set.columns.values.tolist()
l.remove('labels')
#fitting 9 different polynomials
#Example : polynomial of degree 3 is theta(0)X(0) + theta(1)X1 + theta(2)X2 + theta(3)X3
for i in l:
	feature_list.append(i) #USED LIST OF FEATURES GETS INCREASED EVERY ITERATION
	feature = (train_set[feature_list].values) #GETTING VALUES FROM DATA FRAME AS AN ARRAY
	theta = np.ones(len(feature_list)+1) #INITIATING THETA
	a=np.ones(len(feature)) #ADDING X(0)
	X_train_2 = np.insert(feature,0,a,axis=1) #ADDING X(0)
	X_train_2=np.transpose(X_train_2)
	y = np.array(train_set['labels'])
	print('Polynomial of Degree :',len(feature_list))
	params,err_,pred = Gradient_Descent(X_train_2,y,theta)
	train_err.append(err_)
	plt.figure()
	plt.scatter(train_set['feature'],pred)
	plt.xlabel('train_feature')
	plt.ylabel('Prediction')
	plt.show()
#EVALUATING THE TEST ERROR
	feature_test = (test_set[feature_list].values)
	theta = params #PARAMS ARE THE FINAL ITERATION PARAMETERS
	a=np.ones(len(feature_test)) #ADDING X(0)
	X_test = np.insert(feature_test,0,a,axis=1)
	X_test =np.transpose(X_test)
	y_test = np.array(test_set['labels'])
	test_err = np.dot(np.transpose(theta),X_test) - y_test
	err_sq = test_err**2
	cost_test = np.sum(err_sq)
	cost_test = (1/2)*cost_test
	cost_test= cost_test/np.shape(X_test)[1]
	test_err_.append(cost_test)
#PART b
x=[1,2,3,4,5,6,7,8,9]
print('Plot of Degree of Polynomial vs Error')
plt.figure()
plt.plot(x,train_err,color='b')
plt.plot(x,test_err_,color='r')
plt.legend(['Train Error','Test Error'])
plt.xlabel('Degree of Polynomial')
plt.ylabel('Error')
plt.show()
min_err = min(test_err_)  
for i in range(len(test_err_)):
    if test_err_[i] == min_err:
        min_index = i
print('As we see the polynomial best suited is the one with minimum test error test error whose degree is', min_index+1)

