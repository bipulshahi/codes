from sklearn import datasets
digits=datasets.load_digits()
x=digits.data
y=digits.target
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

xtrain=x[:-500]
xtest=x[-500:]
ytrain=y[:-500]
ytest=y[-500:]
print(len(xtrain))
print(len(xtest))
print(len(ytrain))
print(len(ytest))
nn=MLPClassifier(hidden_layer_sizes=(30,30,30),activation='logistic',tol=0.001)
nn.fit(xtrain,ytrain)
print("predicted value")
print(nn.predict(xtest[6]))
img=digits.images
img=img[-500:]
plt.imshow(img[6],cmap="Blues",interpolation="nearest")
plt.show()

input('')



