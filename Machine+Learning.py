
# coding: utf-8

# In[1]:

#Classification
from sklearn.datasets import load_iris
iris=load_iris()


# In[2]:

print(iris.data)

print(iris.target)
print(iris.target_names)


# In[3]:

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)


# In[4]:

knn.fit(iris.data,iris.target)


# In[11]:

knn.predict([[2.2,3,5,1.3],[2,4,7,0.4]])


# In[13]:

iris.target_names[knn.predict([[2.2,3,5,1.3],[2,4,7,0.4]])]


# In[25]:

#Calculate Accuracy
#Comapring Real & Predicted Output
#150- Training & Testing
#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(iris.data,iris.target)


# In[26]:

knn.fit(X_train,Y_train)


# In[27]:

Y_pred=knn.predict(X_test)


# In[30]:

count=0
for i in range (0,38):
    if Y_test[i]==Y_pred[i]:
        count=count+1
print ((count)/38)


# In[29]:

knn.score(X_test,Y_test)


# In[ ]:

#without Using Train Test split Function


# In[34]:

from random import shuffle
shuffle(x)


# In[42]:

X=iris.data
Y=iris.target.reshape(150,1)
#Concate


# In[46]:

import numpy as np
data=np.concatenate((X,Y),axis=1)


# In[48]:

shuffle(data)


# In[49]:

X_train=data[0:100,:4]
X_test=data[100:,:4]
Y_train=data[:100,4:]
Y_test=data[100:,4:]


# In[51]:

knn.fit(X_train,Y_train)


# In[52]:

knn.score(X_test,Y_test)


# In[ ]:



