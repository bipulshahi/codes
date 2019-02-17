
# coding: utf-8

# In[ ]:




# In[ ]:




# In[1]:

import pandas as pd
data=pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)


# In[12]:

X=data[["TV","radio","newspaper"]].as_matrix()
Y=data[["sales"]].as_matrix()
X.shape


# In[13]:

X_train=X[:150,:]
X_test=X[150:,:]

Y_train=Y[:150,:]
Y_test=Y[150:,:]


# In[19]:

#Regression


# In[26]:

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.pairplot(data,x_vars=["TV","radio","newspaper"],y_vars=["sales"],kind='reg',size=7)


# In[27]:

#Sci-Kit learn
from sklearn.linear_model import LinearRegression


# In[28]:

linreg=LinearRegression()


# In[29]:

#.fit---Training using the dataset including features & Labels part
#.predict----predict using features
linreg.fit(X_train,Y_train)


# In[30]:

Y_pred=linreg.predict(X_test)


# In[36]:

import numpy as np
m=np.concatenate((Y_test,Y_pred),axis=1)
m


# In[45]:

#Evaluation Matices
#MAE-Mean Absolute error
#MSE-Mean Squared Error
#RMSE-Root Mean Squared Error
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))


# In[54]:

from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
Y=iris.target


# In[86]:

y=Y.reshape(150,1)
data=np.concatenate((X,y),axis=1)
data
from random import shuffle
shuffle(data)


# In[91]:

data
f=data[:,:4]
l=data[:,4:]
f_train=f[:100,:]
f_test=f[100:,:]

l_train=l[:100,:]
l_test=l[100:,:]


# In[ ]:




# In[96]:

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(f_train,l_train)
y_pred=knn.predict(f_test)
count=0
for i in range(0,50):
    if (y_pred[i]==l_test[i]):
        count=count+1
        
(count/50)*100
        


# In[ ]:



