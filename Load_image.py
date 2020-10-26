#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

DATADIR = "D:\petImages1"
DATADIR1 = "D:\petImages2"
CATEGORIES = ["dogs","cats","panda"]

for category in CATEGORIES:
    path = os.path.join(DATADIR1, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        print(os.path.join(path,img))
        print(img_array.shape)
        plt.imshow(img_array)
        plt.show()
        break
    break


# In[3]:


IMG_SIZE = 150
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array)
print(new_array.shape)
plt.show()


# In[20]:


data = []


def create_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                data.append([new_array,class_num])
            except Exception as e:
                pass
create_data()


# In[4]:


data1 = []
def create_data1():
    for category in CATEGORIES:
        path = os.path.join(DATADIR1, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                data1.append([new_array,class_num])
            except Exception as e:
                pass
create_data1()


# In[5]:


print(len(data))


# In[6]:


print (len(data1))


# In[21]:


import random

random.shuffle(data)
random.shuffle(data1)


# In[22]:


a = []
b = []


# In[9]:


c = []
d = []


# In[23]:




for features, label in data:
    a.append(features)
    b.append(label)

    
#x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    


# In[11]:


for features, label in data1:
    c.append(features)
    d.append(label)

c = np.array(c).reshape(-1,IMG_SIZE,IMG_SIZE,1)


# In[24]:


x = np.array(a).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = []
y = b


# In[25]:


#from sklearn.preprocessing import OneHotEncoder
y = tf.keras.utils.to_categorical(y, num_classes=3)


# In[14]:


d = tf.keras.utils.to_categorical(d, num_classes=3)


# In[15]:


import pickle
pickle_out = open("c.pickle","wb")
pickle.dump(c,pickle_out)
pickle_out.close()

pickle_out = open("d.pickle","wb")
pickle.dump(d,pickle_out)

pickle_out.close()


# In[26]:


x_training = x[0:2600]
y_training = y[0:2600]
len(x_training)


# In[27]:


x_testing = x[2600:]
y_testing = y[2600:]
len(x_testing)


# In[29]:


print(x_training.shape)


# In[28]:


import pickle

pickle_out = open("x_training.pickle","wb")
pickle.dump(x_training,pickle_out)
pickle_out.close()

pickle_out = open("y_training.pickle","wb")
pickle.dump(y_training,pickle_out)

pickle_out = open("x_testing.pickle","wb")
pickle.dump(x_testing,pickle_out)
pickle_out.close()

pickle_out = open("y_testing.pickle","wb")
pickle.dump(y_testing,pickle_out)
pickle_out.close()


# In[ ]:





# In[ ]:





# In[ ]:




