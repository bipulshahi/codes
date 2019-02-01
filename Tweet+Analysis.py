
# coding: utf-8

# In[ ]:

from textblob import TextBlob


# In[ ]:

#hlo=TextBlob("Vipul used to be very angry until someone unnecessary pokes him")
hlo="Vipul used to be very angry until someone unnecessary pokes him"
#hlo.tags


# In[ ]:

hlo.words


# In[ ]:

hlo.sentiment.polarity


# In[ ]:

import tweepy
from textblob import TextBlob

consumer_key='uyguygugu'
consumer_secret='jnjibhj'

access_token='bhjbhjjb-jb jh bjbj'
access_token_secret='n njbjbjknbjkn'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api=tweepy.API(auth)


# In[ ]:

public_tweets=api.search('modi')

for tweet in public_tweets:
    print(tweet.text)
    analysis=TextBlob(tweet.text)
    print(analysis.sentiment)


# In[ ]:

import time
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import re



def calctime(a):
    return time.time()-a

positive=0
negative=0
compound=0

count=0
initime=time.time()
plt.ion()

import test

ckey='Z9lbmPWP4sLOSeXWAEWnE3ipv'
csecret='wBN2vlDlJ2pz2qi9M4gvbFcRDyQjnxx0AVPU4QaGLwH6WbUAlX'
atoken='162988906-MHvzgWxcEFY6WSO9eCd3qe6EPFTy3CL11nZJiMWs'
asecret='sXQiJPqsZEFq2QXuBfJC2WTwHoQxynhznp0gmThtg2CvQ'

class listener(StreamListener):
    
    def on_data(self,data):
        global initime
        t=int(calctime(initime))
        all_data=json.loads(data)
        tweet=all_data["text"]
        #username=all_data["user"]["screen_name"]
        tweet=" ".join(re.findall("[a-zA-Z]+", tweet))
        blob=TextBlob(tweet.strip())

        global positive
        global negative     
        global compound  
        global count
        
        count=count+1
        senti=0
        for sen in blob.sentences:
            senti=senti+sen.sentiment.polarity
            if sen.sentiment.polarity >= 0:
                positive=positive+sen.sentiment.polarity   
            else:
                negative=negative+sen.sentiment.polarity  
        compound=compound+senti        
        print (count)
        print (tweet.strip())
        print (senti)
        print (t)
        print (str(positive) + ' ' + str(negative) + ' ' + str(compound))
        
    
        plt.axis([ 0, 70, -20,20])
        plt.xlabel('Time')
        plt.ylabel('Sentiment')
        plt.plot([t],[positive],'go',[t] ,[negative],'ro',[t],[compound],'bo')
        plt.show()
        plt.pause(0.0001)
        if count==200:
            return False
        else:
            return True
        
    def on_error(self,status):
        print (status)


auth=OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)

api=tweepy.API(auth)

twitterStream=Stream(auth, listener(count))

twitterStream.filter(track=["technology"])


# In[ ]:



