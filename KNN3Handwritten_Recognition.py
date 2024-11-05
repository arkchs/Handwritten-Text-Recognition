#!/usr/bin/env python
# coding: utf-8

# # MNIST Handwritten Digit Recognition

# # Importing libraries 

# In[163]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline


# # Loading the MNIST datasets

# In[164]:


data_df = pd.read_csv("data.csv")
#test_df = pd.read_csv("test.csv")


# In[165]:


data_df.head()


# In[166]:


#test_df.head()


# # For train and test both we will use train.csv (Taking train data as complete data)

# In[167]:


data_df.shape


# # Data Preparation for Model Building

# In[168]:


y=data_df['label']
x=data_df.drop('label',axis=1)


# In[169]:


#x_for_test_data=test_df[:]


# In[170]:


type(x)


# The third line is used so that the row at the some_digit index in the dataset is selected and is then converted into a numpy array.

# In[171]:


plt.figure(figsize=(7,7))
some_digit=2000
some_digit_image = x.iloc[some_digit].to_numpy()
plt.imshow(np.reshape(some_digit_image, (28,28)))
print(y[some_digit])


# In[172]:


sns.countplot( x='label', data=data_df) 


# #### we can conclude that our dataset is balanced

# # Splitting the train data  into train and test 

# In[173]:


from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.75, random_state = 0)


# In[174]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# # **Models**

# # KNN

# In[175]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train,y_train)
# x_train = scaler.transform(x_train)
# x_train.shape


# # k=3

# In[176]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)


# In[177]:


y_pred = classifier.predict(x_test)
y_pred


# In[178]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test, y_pred))


# In[179]:


print(classification_report(y_test, y_pred))


# In[180]:


print(confusion_matrix(y_test, y_pred))


# In[181]:


#y_pred_on_test_data = classifier.predict(x_for_test_data)
#y_pred_on_test_data


# ## **3NN-96.65% accuracy**

# ## **Trying out multiple different values to test the effect of Training:Testing Split and the value of K in KNN has on the model accuracy.**

# ## Note: this method takes a significant amount of time to load. 

# In[1]:


# from sklearn.model_selection import  train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# knn_values = [2,4,5,6,7,10]

# test_split_values=[0.4,0.3,0.25,0.20,0.1,0.05]


# for kv in knn_values:
#     for tsv in test_split_values:
#         print(f"The value of knn classifier is ${kv} and the value of test split is ${tsv}")
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = tsv, random_state = 40)
#         x_train.shape,y_train.shape,x_test.shape,y_test.shape
        
#         classifier = KNeighborsClassifier(n_neighbors = kv)
#         classifier.fit(x_train, y_train)
#         y_pred = classifier.predict(x_test)
#         y_pred
#         print(accuracy_score(y_test, y_pred))
#         print(classification_report(y_test, y_pred))
#         print(confusion_matrix(y_test, y_pred))

