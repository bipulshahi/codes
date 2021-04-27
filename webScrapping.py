#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip3 install bs4')


# In[8]:


#pip install beautifulsoup4
from bs4 import BeautifulSoup as bs
import requests


# In[26]:


link = "https://peopletech01.000webhostapp.com/demo_web.html"


# In[27]:


page = requests.get(link)


# In[28]:


page


# In[29]:


page.content


# In[30]:


#soup = bs(page.content,'html.parser')
soup = bs(page.content)
soup


# In[31]:


print(soup.prettify())


# In[32]:


list(soup.children)


# In[61]:


a = list(soup.children)[2]


# In[62]:


a


# In[64]:


b = list(a.children)[3]
b


# In[67]:


p = list(b.children)[1]


# In[68]:


data = p.get_text()
data


# ## Finding all instances of a tag at once

# In[69]:


#pip install beautifulsoup4
from bs4 import BeautifulSoup as bs
import requests


# In[70]:


link = "https://peopletech01.000webhostapp.com/demo_web2.html"


# In[71]:


page = requests.get(link)


# In[72]:


page


# In[73]:


page.content


# In[79]:


soup = bs(page.content,'html.parser')


# In[80]:


soup


# In[81]:


soup.find_all('p')


# In[82]:


soup.find_all('p')[1].get_text()


# In[83]:


soup.find('p')


# ## Searching the tags by class or id

# In[97]:


link = "https://peopletech01.000webhostapp.com/demo_web3.html"


# In[98]:


page = requests.get(link)


# In[99]:


page


# In[100]:


page.content


# In[101]:


soup = bs(page.content,'html.parser')


# In[102]:


print(soup.prettify)


# In[103]:


soup.find_all('p',class_='outer-text')


# In[104]:


soup.find_all('p',id='first')


# ## Using CSS selectors

# In[105]:


soup.select("div p")


# In[106]:


soup.select("div p.first-item")


# In[107]:


soup.select("div p#first")


# In[108]:


soup.select("body p.outer-text")


# In[109]:


soup.find_all("a")


# In[111]:


soup.find(id="link3")


# In[112]:


my_links = soup.find_all("a")
my_links


# In[113]:


links = []
for link in my_links:
    links.append(link.get('href'))


# In[114]:


links


# ## Amazon product review scrap

# In[115]:


from bs4 import BeautifulSoup as bs
import requests


# In[173]:


link = 'https://www.amazon.in/product-reviews/B089MVKH71'


# In[174]:


page = requests.get(link,'html.parser')


# In[175]:


page


# In[176]:


page.content


# In[184]:


soup = bs(page.content,'html.parser')


# In[185]:


print(soup.prettify())


# In[227]:


names = soup.find_all('span',class_='a-profile-name')


# In[228]:


names


# In[229]:


cust_name = []
for i in range(len(names)):
    cust_name.append(names[i].get_text())
cust_name


# In[230]:


cust_name.pop(0)
cust_name.pop(1)


# In[231]:


cust_name


# In[232]:


len(cust_name)


# In[233]:


#title = soup.find_all('span',class_='review-title')


# In[234]:


#title = soup.find_all('a',class_='review-title-content')
title = soup.find_all('a',class_='review-title-content')


# In[235]:


title


# In[236]:


review_title = []
for i in range(len(title)):
    review_title.append(title[i].get_text())
review_title


# In[237]:


review_title[:] = [titles.lstrip('\n') for titles in review_title]


# In[238]:


review_title[:] = [titles.rstrip('\n') for titles in review_title]


# In[239]:


review_title


# In[240]:


len(review_title)


# In[246]:


rating = soup.find_all('i',class_='review-rating')
rating


# In[247]:


rate = []
for i in range(len(rating)):
    rate.append(rating[i].get_text())
rate


# In[248]:


len(rate)


# In[249]:


rate.pop(0)
rate.pop(1)


# In[250]:


len(rate)


# In[251]:


review = soup.find_all("span",{"data-hook":"review-body"})


# In[252]:


review


# In[253]:


reviews = []
for i in range(len(review)):
    reviews.append(review[i].get_text())
reviews


# In[254]:


len(reviews)


# In[255]:


reviews[:] = [reviews.lstrip('\n') for reviews in reviews]


# In[256]:


reviews[:] = [reviews.rstrip('\n') for reviews in reviews]


# In[257]:


reviews


# In[258]:


len(reviews)


# In[262]:


cust_name
review_title
rate
reviews


# In[263]:


import pandas as pd


# In[264]:


df = pd.DataFrame()
df


# In[265]:


df['Customer Name'] = cust_name


# In[267]:


df['Review Title'] = review_title
df['Ratings'] = rate
df['Reviews'] = reviews


# In[268]:


df.head()


# In[269]:


df.to_csv('Amazon_Reviews.csv')


# In[ ]:




