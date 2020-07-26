#!/usr/bin/env python
# coding: utf-8

# In[31]:


def makedir(directory):
    if not os.path.exists(directory): 
        os.makedirs(directory) 
        return None
    else:
        pass


# In[32]:


import cv2
import os

cap = cv2.VideoCapture(0)

i = 0
image_count = 0

while i < 6:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)

  #ROI
    roi = frame[100:400,320:620]
    cv2.imshow('roi',roi)
    roi = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi , (28,28) , interpolation = cv2.INTER_AREA)

    cv2.imshow('roi scaled and gray' , roi)
    copy = frame.copy()
    cv2.rectangle(copy , (320,100) , (620,400) , (255,0,0) , 5)

    if i == 0:
        image_count = 0
        cv2.putText(copy, 'Press Enter to record 1st object' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
    if i == 1:
        image_count += 1
        cv2.putText(copy, 'Recording 1st object - Train Dataset' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        cv2.putText(copy, str(image_count) , (400,400) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        gesture_one = 'E:/ML_Codes/Mask/train/0/'
        makedir(gesture_one)
        cv2.imwrite(gesture_one + str(image_count) + ".jpg" , roi)
    if i == 2:
        image_count += 1
        cv2.putText(copy, 'Recording 1st object - Test Dataset' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        cv2.putText(copy, str(image_count) , (400,400) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        gesture_one = 'E:/ML_Codes/Mask/test/0/'
        makedir(gesture_one)
        cv2.imwrite(gesture_one + str(image_count) + ".jpg" , roi)
  
    if i == 3:
        image_count = 0
        cv2.putText(copy, 'Press Enter to record 2nd object' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
    if i == 4:
        image_count += 1
        cv2.putText(copy, 'Recording 2nd object - Train Dataset' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        cv2.putText(copy, str(image_count) , (400,400) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        gesture_one = 'E:/ML_Codes/Mask/train/1/'
        makedir(gesture_one)
        cv2.imwrite(gesture_one + str(image_count) + ".jpg" , roi)
    if i == 5:
        image_count += 1
        cv2.putText(copy, 'Recording 2nd object - Test Dataset' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        cv2.putText(copy, str(image_count) , (400,400) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
        gesture_one = 'E:/ML_Codes/Mask/test/1/'
        makedir(gesture_one)
        cv2.imwrite(gesture_one + str(image_count) + ".jpg" , roi)
    if i == 9:
        cv2.putText(copy, 'Hit Enter to Exit' , (100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
    cv2.imshow('frame' , copy)

    if cv2.waitKey(1) == 13:
        image_count = 0
        i += 1
cap.release()
cv2.destroyAllWindows()


# In[33]:


import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras.datasets import cifar10
#from keras import regularizers
#from keras.callbacks import LearningRateScheduler
import numpy as np
import os


# In[34]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu' , input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(1, activation = 'sigmoid'))

print(model.summary())


# In[35]:


import cv2
labels = []
features = []
import os
for i in os.listdir('E:/ML_Codes/Mask/train/0'):
    labels.append(0)
for i in os.listdir('E:/ML_Codes/Mask/train/1'):
    labels.append(1)
    
for i in os.listdir('E:/ML_Codes/Mask/train/0'):
    features.append(cv2.imread(os.path.join('E:/ML_Codes/Mask/train/0',i),0))
for i in os.listdir('E:/ML_Codes/Mask/train/1'):
    features.append(cv2.imread(os.path.join('E:/ML_Codes/Mask/train/1',i),0))


# In[36]:


test_labels = []
test_features = []
import os
for i in os.listdir('E:/ML_Codes/Mask/test/0'):
    test_labels.append(0)
for i in os.listdir('E:/ML_Codes/Mask/test/1'):
    test_labels.append(1)
    
for i in os.listdir('E:/ML_Codes/Mask/test/0'):
    test_features.append(cv2.imread(os.path.join('E:/ML_Codes/Mask/test/0',i),0))
for i in os.listdir('E:/ML_Codes/Mask/test/1'):
    test_features.append(cv2.imread(os.path.join('E:/ML_Codes/Mask/test/1',i),0))


# In[37]:


import numpy as np
features = np.array(features).reshape(-1,28,28,1)
test_features = np.array(test_features).reshape(-1,28,28,1)


# In[38]:


features = features/255
test_features = test_features/255


# In[39]:


labels = np.array(labels)
test_labels = np.array(test_labels)


# In[40]:


#Training your model
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

epochs = 20
batch_size = 32

model.fit(features, labels, batch_size=batch_size, 
          steps_per_epoch=features.shape[0] // batch_size,
          epochs=40,
          verbose=1,validation_data=(test_features,test_labels))


# In[41]:


model.save('Mask.h5')


# In[42]:


from tensorflow.keras.models import load_model
classifier = load_model('Mask.h5')


# In[43]:


def getLetter(result):
    classLabels = {0: 'Normal',
                   1: 'Smiling'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return 'Error'


# In[44]:


#Test your model just bulit
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    frame = cv2.flip(frame,1)

    #region of interest
    roi = frame[100:400 , 220:520]
    cv2.imshow('roi',roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28,28), interpolation = cv2.INTER_AREA)

    #cv2.imshow('roi scaled and gray' , roi)
    copy = frame.copy()
    cv2.rectangle(copy, (220,100) , (520,400) , (255,0,255) , 5)

    roi = roi.reshape(1,28,28,1)
    roi = roi/255
    result = (model.predict(roi) > 0.5).astype("int32")

    cv2.putText(copy,getLetter(result),(150,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow('frame',copy)
    print(result)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




