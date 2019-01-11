import numpy
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
dataset=datasets.load_diabetes()
x=dataset.data
x=x[:,numpy.newaxis,2]
y=dataset.target
xtrain=x[:-30]
xtest=x[-30:]
ytrain=y[:-30]
ytest=y[-30:]
algo=linear_model.LinearRegression()
algo.fit(xtrain,ytrain)
plt.scatter(xtest,ytest,color='red')
yhat=algo.predict(xtest)
#print(yhat)
#print("actual value")
#print(ytest[2:10])
plt.plot(xtest,yhat,color='green',linewidth=2)
plt.show()

