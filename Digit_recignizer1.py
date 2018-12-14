
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[ ]:




# In[ ]:

data=pd.read_csv('E:/ML_Codes/all/train.csv').as_matrix()
data.shape


# In[ ]:

type(data)


# In[ ]:

xtrain=data[0:21000,1:]
train_label=data[0:21000,0]


# In[ ]:

train_label


# In[ ]:

clf=DecisionTreeClassifier()


# In[ ]:

clf.fit(xtrain,train_label)


# In[ ]:

xtest=data[21000:,1:]
actual_label=data[0:21000,0]


# In[ ]:

d=xtest[6]
d.shape=(28,28)
pt.imshow(255-d)

print(clf.predict([xtest[6]]))
pt.show()
         


# In[ ]:

p=clf.predict(xtest)
count=0
for i in range(0,21000):
    count+=1 if p[i]==actual_label[i] else 0
print ("Accuracy=", (count/21000)*100)


# In[ ]:



