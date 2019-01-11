import matplotlib.pyplot as plt
import numpy
from sklearn import datasets, linear_model

bostondata=datasets.load_boston()
x=bostondata.data
y=bostondata.target
import sklearn
##xtrain=x[:-50]
##xtest=x[-50:]
## [1,2,4,5,7,8,5,42,2,1,4,5,4,4,5,2,1,4,5,5,4]
##ytrain=y[:-50]
##ytest=y[-50:]
xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(x,y,test_size=0.08,random_state=3)

alg=linear_model.LinearRegression()
alg.fit(xtrain,ytrain)
testprediction=alg.predict(xtest)
trainprediction=alg.predict(xtrain)
msetest=numpy.mean((ytest-testprediction)**2)
msetrain=numpy.mean((ytrain[:50]-alg.predict(xtrain[:50]))**2)
print("mse for test",msetest,"mse for train",msetrain)
print(alg.predict(xtest[0:5]))
print("actual value")
print(ytest[0:5])
plt.scatter(trainprediction,trainprediction-ytrain,color='blue',alpha=0.5)
plt.scatter(testprediction,testprediction-ytest,color='red',alpha=0.5)
plt.hlines(y=0,xmin=0,xmax=50)
plt.show()



