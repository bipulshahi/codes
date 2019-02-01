
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:

movies_df = pd.read_csv('E:/ML_Codes/ml-1m/ml-1m/movies.dat', sep='::', header=None)


# In[5]:

movies_df.head()


# In[6]:

ratings_df = pd.read_csv('E:/ML_Codes/ml-1m/ml-1m/ratings.dat', sep='::', header=None)


# In[7]:

ratings_df.head()


# In[8]:

movies_df.columns=["MovieID","Title","Genres"]
ratings_df.columns=["UserID","MovieID","Ratings","Timestamp"]


# In[9]:

len(movies_df)
movies_df['List Index']=movies_df.index
movies_df.head()


# In[10]:

merge_df=movies_df.merge(ratings_df, on='MovieID')
merge_df=merge_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)


# In[11]:

merge_df.head(25)


# In[12]:

#group up by user ID
usergroup=merge_df.groupby('UserID')
usergroup.first()


# In[13]:

#formatting the data for RBM
#amount of user used for training
amountOfUsedUsers = 1000
#training list
trx=[]
#for each user in the group
for userID, curUser in usergroup:
    #create a temp to store movie ratings
    temp = [0]*len(movies_df)
    for num, movie in curUser.iterrows():
        #divide the ratings by 5 and store it
        temp[movie['List Index']]=movie['Ratings']/5.0
        
    #now add the list of ratings into training list
    trx.append(temp)
    #check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1


# In[14]:

#Build our RBM with Tensor flow
hiddenUnits = 20
visibleUnits = len(movies_df)
vb = tf.placeholder("float", [visibleUnits]) #number of unique movies
hb = tf.placeholder("float", [hiddenUnits])  #numbers of nurons in hidden layer to learn features
w=tf.placeholder("float", [visibleUnits, hiddenUnits])


# In[15]:

#forward pass or input processing
v0=tf.placeholder("float",[None, visibleUnits])
_h0=tf.nn.sigmoid(tf.matmul(v0, w)+hb)  #probabilities of the hidden units
h0=tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))


# In[16]:

#Backward Pass or Reconstruction
_v1=tf.nn.sigmoid(tf.matmul(h0, tf.transpose(w))+vb)
v1=tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1=tf.nn.sigmoid(tf.matmul(v1, w)+hb)


# In[17]:

#training of a Restricted Boltzman Machine
#train a RBM using Tensorflow and visualize the same
#use stochastic gradient descent to find optimal weight
alpha=1.0
w_pos_grad=tf.matmul(tf.transpose(v0),h0)
w_neg_grad=tf.matmul(tf.transpose(v1),h1)
#contracitive Divergence
CD=(w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
update_w=w+alpha*CD
update_vb=vb + alpha*tf.reduce_mean(v0-v1,0)
update_hb=hb + alpha*tf.reduce_mean(h0-h1,0)

err_sum=tf.reduce_mean(tf.square(v0-v1))


# In[18]:

#Current weight
cur_w=np.zeros([visibleUnits,hiddenUnits],np.float32)
#current visible unit biases
cur_vb=np.zeros([visibleUnits], np.float32)
#current hidden ubit biases
cur_hb=np.zeros([hiddenUnits], np.float32)
#previous weight
prv_w=np.zeros([visibleUnits,hiddenUnits],np.float32)
#previous visible unit biases
prv_vb=np.zeros([visibleUnits], np.float32)
#previous Hidden unit biases
prv_hb=np.zeros([hiddenUnits], np.float32)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


# In[19]:

#We train the RBM with 15 epochs with each epoch using 10 batches
#With size 100.
#After Training , We print out a grph with errors and epoch

epochs = 15
batchsize = 100
errors=[]
for i in range(epochs):
    for start, end in zip(range(0, len(trx), batchsize), range(batchsize, len(trx), batchsize)):
        batch = trx[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, w: prv_w, vb: prv_vb, hb: prv_hb} )
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, w: prv_w, vb: prv_vb, hb: prv_hb} )
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, w: prv_w, vb: prv_vb, hb: prv_hb} )
        prv_w=cur_w
        prv_vb=cur_vb
        prv_hb=cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trx, w: cur_w, vb: cur_vb, hb: cur_hb} ))

plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()


# In[20]:

len(trx[100])


# In[21]:

#Recommendation
#We can predict movies that an arbitray person might like
#This can be done by feeding in the user wathed movie prefrences into RBM and reconstructing the input

#selecting the input user
inputuser = [trx[500]]

#Feeding the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, w)+hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(w))+vb)
feed = sess.run(hh0, feed_dict={v0: inputuser, w:prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0:feed ,w:prv_w ,vb:prv_vb   })


# In[22]:

movies_df["Recommendation Score"] = rec[0]
movies_df.sort_values(["Recommendation Score"], ascending=False).head(5)


# In[23]:

movies_df.head()


# In[ ]:




# In[ ]:



