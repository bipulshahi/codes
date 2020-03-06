
# coding: utf-8

# In[1]:


import cv2


# In[9]:


imagePath='E:/ML_Codes/FaceDetect-master(1)/FaceDetect-master/image.png'
cascPath="E:/ML_Codes/FaceDetect-master(1)/FaceDetect-master/haarcascade_frontalface_default.xml"


# In[10]:


#Read the image
image=cv2.imread(imagePath)


# In[12]:


#Create a haar cascade
face=cv2.CascadeClassifier(cascPath)


# In[14]:


myfaces=face.detectMultiScale(image,
                             minSize=(35,35),
                             scaleFactor=1.1,
                             minNeighbors=5,
                             flags=cv2.CASCADE_SCALE_IMAGE)


# In[19]:


for (x,y,w,h) in myfaces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)


# In[25]:


cv2.imshow("Faces Found "+str(len(myfaces)), image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[30]:





# In[31]:


import cv2
im=cv2.VideoCapture(0)
state,frame=im.read()
cv2.imshow('my_image',frame)
cv2.waitKey(0)
im.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
video=cv2.VideoCapture(0)
a=1
while True:
    a=a+1
    check, frame = video.read()
    #print(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('capturing',frame)
    key=cv2.waitKey(1)
    if key == ord('a'):
        break
        
print(a)
video.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
cascpath='E:/ML_Codes/FaceDetect-master(1)/FaceDetect-master/haarcascade_frontalface_default.xml'
faceCascade=cv2.CascadeClassifier(cascpath)
webcam = cv2.VideoCapture(0)  
(_, im) = webcam.read()
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(30,30),
                                  flags=cv2.CASCADE_SCALE_IMAGE)
print(len(faces))
for (x ,y, w, h) in faces:
    cv2.rectangle(im, (x,y), 
                  (x+w, y+h), (0,255,0),2)

cv2.imshow('Faces Found', im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # NLP=Natural Language Processing

# In[ ]:


1. NLTK - Natural language tool kit
2. textblob
3. tweepy


# In[32]:


import nltk


# In[33]:


nltk.download()


# In[ ]:




