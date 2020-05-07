import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
thresh_hold = 0.000000001#Adjust this for faster processing !
def Gradient_Descent(X_train,y,theta):
	err = np.dot(np.transpose(theta),X_train) - y
	err_sq = err**2
	cost = np.sum(err_sq)
	cost = (1/2)*cost
	cost= cost/np.shape(X_train)[1]
	print('Starting Cost theta initiated as one:',cost)
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
#This is Lasso Regularization
def grad_descent_reg_1(X_train,y,lambda_):
	theta = np.ones(np.shape(X_train)[0])
	err = np.dot(np.transpose(theta),X_train) - y
	err_sq = err**2
	sum_weights= np.sum(theta)
	cost = np.sum(err_sq) + sum_weights
	cost = (1/2)*cost
	cost= cost/np.shape(X_train)[1]
	old = cost
	diff = cost
	reg_lamda = lambda_*np.ones(np.shape(X_train)[0])
	while(diff >= thresh_hold):
		grad = np.dot(X_train,err)+reg_lamda
		grad = grad/np.shape(X_train)[1]
		alpha = 0.05
		theta = theta - alpha*grad
		pred = np.dot(np.transpose(theta),X_train)
		err =  pred - y
		err_sq = err**2
		sum_weights= np.sum(theta)
		cost = np.sum(err_sq) + sum_weights
		cost = (1/2)*cost
		cost= cost/np.shape(X_train)[1]
#		print('Iter_Cost: ',cost)
		diff = abs(cost-old)
		old = cost
	print('Final Lasso Train cost::',cost)
	return(theta,cost,pred)

#This is ridge regularization
def grad_descent_reg_2(X_train,y,lambda_):
	theta = np.ones(np.shape(X_train)[0])
	err = np.dot(np.transpose(theta),X_train) - y
	err_sq = err**2
	sum_sq_weights = np.sum(np.square(theta))
	cost = np.sum(err_sq)+sum_sq_weights
	cost = (1/2)*cost
	cost= cost/np.shape(X_train)[1]
	old = cost
	diff = cost
	reg_lamda = lambda_*theta
	while(diff >= thresh_hold):
		grad = np.dot(X_train,err)+2*reg_lamda
		grad = grad/np.shape(X_train)[1]
		alpha = 0.05
		theta = theta - alpha*grad
		pred = np.dot(np.transpose(theta),X_train)
		err =  pred - y
		err_sq = err**2
		sum_sq_weights = np.sum(np.square(theta))
		cost = np.sum(err_sq)+sum_sq_weights
		cost = (1/2)*cost
		cost= cost/np.shape(X_train)[1]
#		print('Iter_Cost: ',cost)
		diff = abs(cost-old)
		old = cost
	print('Final Ridge Train cost::',cost)
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

min_err = min(train_err)  
for i in range(len(train_err)):
    if train_err[i] == min_err:
        min_index = i
max_err = max(train_err)  
for i in range(len(train_err)):
    if train_err[i] == max_err:
        max_index = i
feature_list = []
min_c_r1_train=[]
min_c_r2_train=[]
min_c_r1_test=[]
min_c_r2_test=[]
max_c_r1_train=[]
max_c_r2_train=[]
max_c_r1_test=[]
max_c_r2_test=[]
print('\n')
print('Performing regularization on minimum train cost polynomial')
print('\n')
print('Minimum error:',min_err)
print('\n')
for i in range(min_index+1):
	feature_list.append(l[i]) #USED LIST OF FEATURES GETS INCREASED EVERY ITERATION
feature = (train_set[feature_list].values) #GETTING VALUES FROM DATA FRAME AS AN ARRAY
theta = np.ones(len(feature_list)+1) #INITIATING THETA
a=np.ones(len(feature)) #ADDING X(0)
X_train_2 = np.insert(feature,0,a,axis=1) #ADDING X(0)
X_train_2=np.transpose(X_train_2) 
y = np.array(train_set['labels'])
l_ = [0.25,0.5,0.75,1]
for i in l_:
	print('Lambda:',i)
	params,cost_train,pred = grad_descent_reg_1(X_train_2,y,i)
	min_c_r1_train.append(cost_train)    
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
	min_c_r1_test.append(cost_test)
	params,cost_train,pred = grad_descent_reg_2(X_train_2,y,i)
	min_c_r2_train.append(cost_train)
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
	min_c_r2_test.append(cost_test)
print('\n')
print('Performing regularization on maximum train cost polynomial')
print('\n')
print('Maximum error:',max_err)
print('\n')
feature_list = []
for i in range(max_index+1):
	feature_list.append(l[i]) #USED LIST OF FEATURES GETS INCREASED EVERY ITERATION

feature = (train_set[feature_list].values) #GETTING VALUES FROM DATA FRAME AS AN ARRAY
theta = np.ones(len(feature_list)+1) #INITIATING THETA
a=np.ones(len(feature)) #ADDING X(0)
X_train_2 = np.insert(feature,0,a,axis=1) #ADDING X(0)
X_train_2=np.transpose(X_train_2)
y = np.array(train_set['labels'])
for i in l_:
	print('Lambda: ',i)    
	params,cost_train,pred = grad_descent_reg_1(X_train_2,y,i)
	max_c_r1_train.append(cost_train)    
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
	max_c_r1_test.append(cost_test)
	params,cost_train,pred = grad_descent_reg_2(X_train_2,y,i)
	max_c_r2_train.append(cost_train)
	feature_test = (test_set[feature_list].values)
	theta = params #PARAMS ARE THE FINAL ITERATION PARAMETERS
	a=np.ones(len(feature_test)) #ADDING X(0)
	X_test = np.insert(feature_test,0,a,axis=1)
	X_test =np.transpose(X_test)
	y_test = np.transpose(test_set['labels'])
	test_err = np.dot(np.transpose(theta),X_test) - y_test
	err_sq = test_err**2
	cost_test = np.sum(err_sq)
	cost_test = (1/2)*cost_test
	cost_test= cost_test/np.shape(X_test)[1]
	max_c_r2_test.append(cost_test)
    
#plotting graph for Lasso regulariztion
print('\n')
print('Plots for minimum train cost polynomial')
plt.figure()
plt.plot(l_,min_c_r1_test,c='r')
plt.plot(l_,min_c_r1_train,c='b')
plt.legend(['test','train'])
plt.title('Lasso regularization')
plt.show()
plt.figure()
plt.plot(l_,min_c_r2_test,c='r')
plt.plot(l_,min_c_r2_train,c='b')
plt.legend(['test','train'])
plt.title('Ridge regularization')
plt.show()
print('\n')
#plotting graph for ridge regularization
print('Plots for maximum train cost polynomial')
plt.figure()
plt.plot(l_,max_c_r1_test,c='r')
plt.plot(l_,max_c_r1_train,c='b')
plt.legend(['test','train'])
plt.title('Lasso regularization')
plt.show()
plt.figure()
plt.plot(l_,max_c_r2_test,c='r')
plt.plot(l_,max_c_r2_train,c='b')
plt.legend(['test','train'])
plt.title('Ridge regularization')
plt.show()
print('From the plots we see that only minimum train cost is significantly affected by regularization(as expected) and since our motive was to increase bias, we hope to get high train error, from the plots hence Ridge regression seems to do a better job')
print('We also see that as lambda increases the Lasso regression train cost keeps on decreasing while the ridge regression train cost although constant is much higher than the original minimum error,indicating that Lasso might still overfit the data')
print('I will use Ridge regression for this problem')