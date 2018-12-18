
# coding: utf-8

# In[1]:

import tensorflow as tf

mnist=tf.keras.datasets.mnist  #28X28 handwritten 0-9

(x_train,y_train),(x_test,y_test)=mnist.load_data()

#import matplotlib.pyplot as plt
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show
#print(x_train[0])

#normalize to scale

x_train=tf.keras.utils.normalize(x_train)
x_test=tf.keras.utils.normalize(x_test)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)


# In[13]:

val_loss, val_acc=model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[10]:

import matplotlib.pyplot as plt
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()
#print(x_train[0])

#model.save('epic_num_reader.model')
#new_model=tf.keras.models.load_model('epic_num_reader.model')
pred=model.predict([x_test])
print(pred)


# In[25]:

import numpy as np
print(np.argmax(pred[7]))


# In[26]:

plt.imshow(x_test[7])
plt.show()


# In[2]:

type(pred)


# In[ ]:



