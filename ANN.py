
# coding: utf-8

# In[128]:


#Feature Extraction of Image

Train_dir='E:/ML_Codes/all1/train'
Test_dir='E:/ML_Codes/all1/test'
IMG_SIZE=50


# In[129]:


#Label Extraction
def label_img(img):
    img_label=img.split('.')[0]
    if img_label=='cat':
        return 0
    elif img_label == 'dog':
        return 1


# In[160]:


#Feature Extraction
import numpy as np
import os
#import tqdm
import cv2
from random import shuffle
def create_train_data():
    training_data=[]
    for img in os.listdir(Train_dir):
        label=label_img(img)
        path=os.path.join(Train_dir,img)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([img, label])
    shuffle(training_data)
    return training_data


# In[161]:


train=create_train_data()


# In[ ]:





# In[15]:


def create_test_data():
    testing_data=[]
    for img in os.listdir(Test_dir):
        path=os.path.join(Test_dir,img)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img)])
    shuffle(testing_data)
    return testing_data


# In[16]:


test=create_test_data()


# In[17]:


test[0]


# In[134]:


import tensorflow as tf
#conda install -c conda-forge tensorflow


# In[162]:


np.array(train).shape


# In[163]:


train_ar=np.zeros((25000,2501))
for i in range(25000):
    train_ar[i]=np.hstack((np.array(train[i])[0].ravel(),
                           np.array(train[i])[1]))


# In[164]:


X=train_ar[:,:-1]
Y=train_ar[:,-1]


# In[167]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y)


# In[126]:


t.shape


# In[ ]:





# In[104]:


feature=t[:,:1]
label=t[:,1:]


# In[118]:


feature .shape


# In[107]:


label


# In[108]:


f_train=tf.keras.utils.normalize(feature)


# In[91]:


#Create Neural Structure
model=tf.keras.models.Sequential()
#Input Layer
model.add(tf.keras.layers.Flatten())
#First Hidden Layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#Second Hidden Layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#output Layer
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))


# In[92]:


#Error Optimization
model.compile(optimizer= 'adam' , loss= 'sparse_categorical_crossentropy' , 
              metrics=['accuracy'])


# In[93]:


model.fit(feature,label,epochs=3)


# In[ ]:




