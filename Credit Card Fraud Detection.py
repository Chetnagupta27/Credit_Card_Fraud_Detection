#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


#loading the dataset to pandas dataframe
credit_card_data=pd.read_csv('creditcard.csv')


# In[6]:


#tail gives last five rows of data
credit_card_data.tail()


# In[7]:


#dataset information
credit_card_data.info()


# In[8]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[10]:


# distribution of legit transaction and fraudulant transaction
credit_card_data['Class'].value_counts()

# here 0 defines legit transaction 
# and 1 defines fraud transaction


# In[11]:


# Seprating data for analysis
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


# In[13]:


legit.head()


# In[14]:


print(legit.shape)
print(fraud.shape)


# In[15]:


#Statistical measure of data
legit.Amount.describe()


# In[16]:


fraud.Amount.describe()


# In[17]:


# compare the values for both the classes 
credit_card_data.groupby('Class').mean()


# Under_Sampling

# Build a simple dataset containing similar distribution of normal transaction and Fraudulent Transaction

# no. of fraudulant transactions --> 492

# In[19]:


legit_sample=legit.sample(n=492)


# Concatenating two dataFrames

# In[20]:


new_dataset=pd.concat([legit_sample, fraud], axis=0)


# axis = 0 means row wise and axis=1 means column wise

# In[21]:


new_dataset.head()


# In[22]:


new_dataset.tail()


# In[23]:


new_dataset['Class'].value_counts()


# In[24]:


new_dataset.groupby('Class').mean()


# Split the data into Features and Targets

# In[25]:


X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']


# In[26]:


print(X)


# In[27]:


print(Y)


# Split the data into training and testing data

# In[28]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2)


# In[30]:


print(X_train)
print("*****************************")
print(X_test)
print("*****************************")
print(Y_train)
print("*****************************")
print(Y_test)


# Model Training

# Logistic Regression

# In[31]:


model=LogisticRegression()


# In[32]:


#training the logistic regression model with Training data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[33]:


#accuracy on training data
X_train_predictions=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_predictions,Y_train)


# In[34]:


print("Accuracy on training data : ", training_data_accuracy)


# In[35]:


#Accuracy on test data
x_test_predictions=model.predict(X_test)
test_data_accuracy=accuracy_score(x_test_predictions,Y_test)


# In[36]:


print("Accuracy on test data : ", test_data_accuracy)


# if acc. score of test data and train data then our model is either underfit or overfit

# In[ ]:




