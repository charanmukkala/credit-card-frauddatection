#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


credit_card_data = pd.read_csv('creditcard.csv',sep=',')


# In[6]:


credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[4]:


# dataset informations
credit_card_data.info()


# In[6]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[5]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[6]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[7]:


print(legit.shape)
print(fraud.shape)


# In[8]:


# statistical measures of the data
legit.Amount.describe()


# In[9]:


fraud.Amount.describe()


# In[10]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[11]:


legit_sample = legit.sample(n=492)


# In[12]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[13]:


new_dataset.head()


# In[14]:


new_dataset.tail()


# In[15]:


new_dataset['Class'].value_counts()


# In[16]:


new_dataset.groupby('Class').mean()


# In[17]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


model = LogisticRegression()


# In[23]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[24]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[25]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[26]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[27]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:




