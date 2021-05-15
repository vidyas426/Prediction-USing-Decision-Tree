#!/usr/bin/env python
# coding: utf-8

# Datascience and Business Analytics
# Task-2
# Prediction using Decision Tree Algorithm
# BY,
# VIDYA TS

# Import Necessary Liraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


# Read csv file using pandas

# In[2]:


dataset = pd.read_csv('Iris.csv')
dataset.head()


# In[3]:


dataset['Species'] = dataset['Species'].map({'Iris-virginica': 0,'Iris-versicolor':1,'Iris-setosa':2})


# In[4]:


dataset.drop('Id',axis=1,inplace=True)


# Splitting the dataset into training and testing 

# In[5]:


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101) 


# Standardization

# In[7]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# Decision Tree Algorithm

# In[8]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini',max_features='auto',splitter='best')
model.fit(x_train,y_train)


# Cross Validation

# In[9]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=model, X=x_train ,y=y_train,cv=10)
print("accuracy is {:.2f} %".format(accuracies.mean()*100))
print("std is {:.2f} %".format(accuracies.std()*100))


# Model Prediction on testing dataset

# In[10]:


predictions = model.predict(x_test)


# In[11]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# Accuracy obtained on the test dataset

# In[12]:


print(accuracy_score(predictions,y_test))


# Decision Tree Visualization

# In[35]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model,filled=True)


# In[36]:


print(tree.export_text(model))


# Prediction On new datapoints

# In[42]:


new_pre=model.predict([[4.5,7.5,8.5,6.5]])
if new_pre == [0]:
    print('Iris-virginica')
elif new_pre == [1]:
    print('Iris_versicolor')
else:
    print('Iris_setosa')


# In[ ]:





# In[ ]:




