#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn 
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)


# In[2]:


bos.head()


# In[4]:


bos.info()


# In[5]:


print(boston.data.shape)
print(boston.feature_names)


# In[6]:


bos.columns = boston.feature_names
print(bos.head())
print(boston.target.shape)


# In[7]:


bos["PRICE"] = boston.target
print(bos.head())
print(bos.describe())


# In[8]:


X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[9]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
Y_pred = lm.predict(X_test)
plt.scatter(Y_test , Y_pred , color = "purple")
plt.xlabel("Prices:Y")
plt.ylabel("Predicted prices : Y1")
plt.title("prices vs Predicted prices: Y vs Y1")
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)

