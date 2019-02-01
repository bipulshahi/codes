
# coding: utf-8

# In[48]:


import os
import nltk
import nltk.corpus


# In[49]:


print(os.listdir(nltk.data.find('corpora')))


# In[5]:


from nltk.corpus import brown
brown.words()


# In[6]:


nltk.corpus.gutenberg.fileids()


# In[7]:


hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet


# In[21]:


for word in hamlet[:500]:
    print(word, end='_')


# In[17]:


#word_tokenize


# In[23]:


AI = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."


# In[24]:


from nltk.tokenize import sent_tokenize, word_tokenize
AI_token=word_tokenize(AI)
AI_token


# In[ ]:


len(AI_token)


# In[22]:


from nltk.probability import FreqDist
fdist=FreqDist()


# In[23]:


for word in AI_token:
    fdist[word.lower()]+=1
fdist


# In[ ]:


fdist['you']


# In[ ]:


len(fdist)


# In[24]:


fdist_top10=fdist.most_common(10)
fdist_top10


# In[28]:


from nltk.tokenize import blankline_tokenize
AI_blank=blankline_tokenize(AI)
AI_blank


# In[27]:


AI_blank[0]


# In[ ]:


#Three types of tokens we have:-
#Biagrams:- Tokens of two consecutive written words known as bigram
#Trigram:-Token of three consecutive written words known as Trigram
#Ngram:-Token of any number of consecutive words known as Ngram


# In[ ]:


from nltk.util import bigrams, trigrams, ngrams


# In[30]:


string='The best and most beautiful things in the world cannot be seen or even touched, they must be felt with the heart'
quotes_tokens=nltk.word_tokenize(string)
quotes_tokens


# In[31]:


quotes_bigrams=list(nltk.bigrams(quotes_tokens))
quotes_bigrams


# In[ ]:


quotes_trigrams=list(nltk.trigrams(quotes_tokens))
quotes_trigrams


# In[ ]:


quotes_ngrams=list(nltk.ngrams(quotes_tokens, 5))
quotes_ngrams


# In[ ]:


#Stemings:-Changes to the token
#Normalize words into its base form or root form
#Affection, Affects, Affections, Affected, Affection, Affecting
#Affect
#Steming algorith works by cutting ends or beginning of the words
#taking into account the most common word


# In[1]:


from nltk.stem import PorterStemmer
pst=PorterStemmer()
pst.stem('having')


# In[4]:


words_to_stem=['give','giving','given','gave']
for words in words_to_stem:
    print(words+":"+pst.stem(words))


# In[5]:


from nltk.stem import LancasterStemmer
lst=LancasterStemmer()
for words in words_to_stem:
    print(words+":"+lst.stem(words))


# In[ ]:


#we can conclude LST is more aggresibe then PST


# In[6]:


from nltk.stem import SnowballStemmer
sbst=SnowballStemmer('english')
for words in words_to_stem:
    print(words+":"+sbst.stem(words))


# In[ ]:


#lemmatization:-In case when steming dont results correctly for ex. Fish, Fishes, Fisherman. It uses
#morphological analysis of words


# In[ ]:


#What does Lemmmatization do?
#1.Groups Together different inflected forms of a word, called lemma
#2.Somehow similar to stemming, as it maps several word into a common root
#3.Output of Lemmatization is a proper word
#4.For example, a lemmatiser should map gone, going and went into go


# In[8]:


from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_len=WordNetLemmatizer()
word_len.lemmatize('corpora')


# In[15]:


words_to_lem=['gone', 'Going']
for words in words_to_lem:
    print(words+":"+word_len.lemmatize(words))


# In[ ]:


#Stop_Words
#several words in english such as I,at,for,begin,got,know,various
#which are usefull in making sentences but these are not usefull in NLP
#So these are called Stop Words


# In[16]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[17]:


len(stopwords.words('english'))


# In[18]:


fdist_top10


# In[25]:


import re
punctuation=re.compile(r'[-.?!,:;()\'|0-9]')
#compile function from re module to form a list with any digit or special charater


# In[30]:


post_punctuation=[]
for words in AI_token:
    word=punctuation.sub("/",words)
    if len(word)>0:
        post_punctuation.append(word)


# In[31]:


post_punctuation


# In[ ]:


AI_token


# In[ ]:


#POS: Parts of speech
#Generally speaking gramatical types of words like noun,verb,adverb, ajectives
#a word can have more then one parts of speech based on the context it is used
#Ex. "Google something on the internet", Here google is used as a verb although its a noun. 
#these are some sort of ambiquities or difficulties which makes the NLU much more harder as compared to NLG
#Because once u understand the language Generation is quite easy


# In[ ]:


#POS tags are used to describe whether a word is noun, an adjective, a proper noun, sigular, plural, verb, adverb, symbol


# In[ ]:


sent='krishna is a natural when it comes to singing'
sent_tokens=word_tokenize(sent)


# In[ ]:


for token in sent_tokens:
    print(nltk.pos_tag([token]))


# In[42]:


sent2='Vipul is walking through the park'
sent2_tokens=word_tokenize(sent2)
for token in sent2_tokens:
    print(nltk.pos_tag([token]))


# In[ ]:


#named Entity Recognition
#naming such as--
#movie
#monetary value
#organization
#location
#Quantities
#person from a text


# In[ ]:


#Three phases of NAE
#1. Noun Pharase identification:-Extract all the noun phrases from a text using dependency passing and parts of speech tagging
#2. phase classification:- in this step all the extracted nouns and phrases are classified into respective categories such as location names and much more
#3. Validation layer to evalute if something goes wrong using knowledge graphs
#popular knowledge craft:- Google knowledge graph, IBM Watson, Wkipedia
#Google's CEO Sundar Pichai introduced the new pixel at Minnesota Roi Centre Event
#Google-Organization
#Sundar Pichai- person
#Minnesota-Location
#Roi centre event-location


# In[36]:


import nltk
from nltk import ne_chunk
NE_sent="The Indian Politicians shouts in the Parliament House"


# In[37]:


NE_tokens=word_tokenize(NE_sent)
NE_tags=nltk.pos_tag(NE_tokens)


# In[38]:


NE_NER=ne_chunk(NE_tags)
print(NE_NER)


# In[ ]:


#NER Entities LIST
#Facility
#Location
#Organization
#Person
#Geo-Socio-political Group
#Geo-Political Entity
#Facility


# In[ ]:


#syntax:- Linguistics syntax is a set of rules, priciples and the processes that govern the structure of a sentence in a given language
#The term syntax is also used to refer to the study of such principles and processes
#So we have certain rule regarding what part of sentence should come up on what positions
#with these rules we create a syntax tree whenever there is a sentence as an input


# In[ ]:


#Syntax tree is a tree representation of syntactic structure of sentences or strings


# In[ ]:


#chunking:- Picking up individual pieces of information and grouping them into bigger pieces


# In[39]:


#WE caught the pink Panther
#The big cat ate the little mouse who was after fresh cheese
new="The big cat ate the little mouse who was after fresh cheese"
new_tokens=nltk.pos_tag(word_tokenize(new))
new_tokens


# In[44]:


grammar_np=r"NP: {<DT><JJ><NN>}"


# In[45]:


chunk_parser=nltk.RegexpParser(grammar_np)


# In[46]:


chunk_result=chunk_parser.parse(new_tokens)


# In[47]:


chunk_result


# In[ ]:




