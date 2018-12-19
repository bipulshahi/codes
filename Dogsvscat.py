
# coding: utf-8

# In[1]:


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm 

TRAIN_DIR = 'E:/ML_Codes/all1/train'
TEST_DIR = 'E:/ML_Codes/all1/test'
IMG_SIZE = 50
LR = 0.0003

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


# In[2]:


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very dog]
    elif word_label == 'dog': return [0,1]


# In[3]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# In[4]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# In[5]:


train_data = create_train_data()


# In[6]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, 
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)


# In[7]:


train = train_data[:-500]
test = train_data[-500:]


# In[8]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


# In[9]:


model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), show_metric=True, run_id=MODEL_NAME)


# In[13]:


import matplotlib.pyplot as plt

test_data = process_test_data()

#test_data=np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    #cat: [1,0]
    #dog:[0,1]
    
    img_data = data[0]
    
    y=fig.add_subplot(3,4,num+1)
    
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE,1)
    
    model_out=model.predict([data])[0]
    
    if np.argmax(model_out) == 1:
        str_label='DOG'
    else:
        str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    
plt.show()
    
    
    


# In[10]:





# In[ ]:




