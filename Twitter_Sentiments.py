
# coding: utf-8

# In[1]:

from textblob import TextBlob


# In[2]:

hlo=TextBlob("Vipul used to be very happy until someone unnecessary pokes him")
hlo.tags


# In[3]:

hlo.words


# In[4]:

hlo.polarity


# In[16]:

import tweepy
from textblob import TextBlob

consumer_key='nm m nm '
consumer_secret='b jb jhj'

access_token='n n n n'
access_token_secret='nmknknk'

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api=tweepy.API(auth)



# In[18]:

public_tweets=api.search('Satue of Unity')

for tweet in public_tweets:
    print(tweet.text)
    analysis=TextBlob(tweet.text)
    print(analysis.sentiment)


# In[ ]:



