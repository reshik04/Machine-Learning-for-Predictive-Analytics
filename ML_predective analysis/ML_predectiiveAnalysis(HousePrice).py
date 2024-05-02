#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install seaborn')

import pandas as pd 
import seaborn as sns
import numpy as np 


# In[3]:


data = pd.read_csv("D:\AI master class\house_data\house.csv")


# In[4]:


data.head()


# In[5]:


data.columns


# In[7]:


data.shape 


# In[8]:


data.describe()


# In[9]:


#visualisation
data.isnull().sum()


# In[11]:


#visualisation 
data.head()
sns.relplot(x='Price',y='number of bedrooms', data = data )


# In[12]:


sns.relplot(x='Price',y='number of bathrooms', data = data )


# In[16]:


sns.relplot(x='Price',y='Area of the house(excluding basement)',hue='waterfront present',data=data )


# In[18]:


#model 
data.head()


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[24]:


train = data.drop(['Price','id','Date'],axis=1)
test = data['Price']


# In[26]:


X_train, X_test,Y_train,Y_test = train_test_split(train, test, test_size=0.3, random_state = 2)


# In[27]:


regr = LinearRegression()


# In[28]:


regr.fit(X_train,Y_train)


# In[30]:


pred = regr.predict(X_test)


# In[31]:


pred


# In[32]:


regr.score(X_test,Y_test)


# In[ ]:




