
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


movies_df = pd.read_csv('E:/ML_Codes/ml-1m/ml-1m/movies.dat', sep='::', header=None)


# In[3]:


ratings_df = pd.read_csv('E:/ML_Codes/ml-1m/ml-1m/ratings.dat', sep='::', header=None)


# In[4]:


movies_df.columns=["MovieID","Title","Genres"]
ratings_df.columns=["UserID","MovieID","Ratings","Timestamp"]
len(ratings_df)


# In[5]:


len(movies_df)
movies_df['List Index']=movies_df.index
movies_df


# In[6]:


merge_df=movies_df.merge(ratings_df, on='MovieID')
merge_df=merge_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)


# In[7]:


#group up by user ID
usergroup=merge_df.groupby('UserID')
usergroup.first()


# In[8]:


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


# In[9]:


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


# In[10]:


#Build our RBM with Tensor flow
hiddenUnits = 20
visibleUnits = len(movies_df)
vb = tf.placeholder("float", [visibleUnits]) #number of unique movies
hb = tf.placeholder("float", [hiddenUnits])  #numbers of nurons in hidden layer to learn features
w=tf.placeholder("float", [visibleUnits, hiddenUnits])


# In[11]:


#forward pass or input processing
v0=tf.placeholder("float",[None, visibleUnits])
_h0=tf.nn.sigmoid(tf.matmul(v0,w)+hb) #probabilities of the hidden unit
h0=tf.nn.relu(tf.sign(_h0-tf.random_uniform(tf.shape(_h0))))


# In[12]:


#backward Pass or reconstruction
_v1=tf.nn.sigmoid(tf.matmul(h0,tf.transpose(w))+vb)
v1=tf.nn.relu(tf.sign(_v1-tf.random_uniform(tf.shape(_v1))))
h1=tf.nn.sigmoid(tf.matmul(v1,w)+hb)


# In[32]:


#training of a RBM
#train a RBM using Tensorflow and visualize the same
#use gradient descent to find optimal weight
alpha=0.1
w_pos_grad=tf.matmul(tf.transpose(v0),h0)
w_neg_grad=tf.matmul(tf.transpose(v1),h1)

#contracitive Divergence
CD=(w_pos_grad-w_neg_grad)/tf.to_float(tf.shape(v0[0]))
update_w=w+alpha*CD
update_vb=vb+alpha*tf.reduce_mean(v0-v1,0)
update_hb=hb+alpha*tf.reduce_mean(h0-h1,0)
err_sum=tf.reduce_mean(tf.square(v0-v1))


# In[33]:


#current weight
curr_w=np.zeros([visibleUnits,hiddenUnits],np.float32)
#Crrent visible unit biases
curr_vb=np.zeros([visibleUnits],np.float32)
#Current hidden unit biases
curr_hb=np.zeros([hiddenUnits], np.float32)
#previous weight
prv_w=np.zeros([visibleUnits,hiddenUnits], np.float32)
#previous visible unit biases
prv_vb=np.zeros([visibleUnits], np.float32)
#previous hidden unit biases
prv_hb=np.zeros([hiddenUnits],np.float32)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


# In[38]:


#we train the RBM with 15 epochs with each epoch using 
#10 batches
#With Size 100
#After training, We generate a graph with errors vs epoch
epoch=20
batchsize=100
errors=[]
for i in range(epoch):
    for start, end in zip(range(0, len(trx), batchsize), 
                          range(batchsize, len(trx), batchsize)):
        batch=trx[start:end]
        curr_w=sess.run(update_w,feed_dict={v0:batch, w:prv_w, vb:prv_vb, hb:prv_hb})
        curr_vb=sess.run(update_vb,feed_dict={v0:batch, w:prv_w, vb:prv_vb, hb:prv_hb})
        curr_hb=sess.run(update_hb,feed_dict={v0:batch, w:prv_w, vb:prv_vb, hb:prv_hb})
        prv_w=curr_w
        prv_vb=curr_vb
        prv_hb=curr_hb
    errors.append(sess.run(err_sum, feed_dict={v0:trx, w:curr_w, vb:curr_vb, hb:curr_hb }))
    
plt.plot(errors)
plt.ylabel('errors')
plt.xlabel('epochs')
plt.show()
    
    
    
    
    
    
    


# In[40]:


len(trx[200])


# In[51]:


#Recommendation
#We can predict movies that an arbitary person might like
#This can be done by feeding in the user watched movie 
#prefrences into RBM and reconstructing the input

#select the user
inputuser = [trx[50]]

#Feed the user and resconstruct the input
hh0=tf.nn.sigmoid(tf.matmul(v0,w)+hb)
vv1=tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(w))+vb)
feed=sess.run(hh0, feed_dict={v0:inputuser,w:prv_w,hb:prv_hb})
rec = sess.run(vv1, feed_dict={hh0:feed,w:prv_w,vb:prv_vb})


# In[52]:


movies_df["Recommendation Score"]=rec[0]
movies_df.sort_values(["Recommendation Score"],ascending=False).head(10)


# In[ ]:




