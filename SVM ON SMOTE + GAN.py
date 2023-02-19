#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics


# In[2]:


Bearing_Data_Expanded = pd.read_csv('Bearing_Expanded_Dataset.csv')
Bearing_Data_Expanded['Misaligned'] = Bearing_Data_Expanded['Misaligned'].astype(int)


# In[3]:


## SVM


# In[4]:


## Train-test split


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Bearing_Data_Expanded[['f','e','hmin0','hminL']], Bearing_Data_Expanded['Misaligned'], test_size=0.3, random_state=42)


# In[7]:


## SVC Training


# In[8]:


from sklearn.svm import SVC


# In[9]:


model = SVC()


# In[10]:


model.fit(X_train,y_train)


# In[11]:


## Predictions + Evaluations


# In[12]:


predictions = model.predict(X_test)


# In[13]:


from sklearn.metrics import classification_report,confusion_matrix


# In[14]:


cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['0', '1'], yticklabels=['0', '1'],
       xlabel='Predicted label', ylabel='True label')
ax.set_ylim(len(cm) - 0.5, -0.5)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")
fig.tight_layout()
plt.show()


# In[15]:


target_names = ['Aligned', 'Misaligned']
report = classification_report(y_test, predictions, target_names=target_names)
print(report)


# In[16]:


## Gridsearch (Finding the right parameters )


# In[17]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[18]:


from sklearn.model_selection import GridSearchCV


# In[19]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[20]:


grid.fit(X_train,y_train)


# In[21]:


grid.best_params_


# In[22]:


grid_predictions = grid.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix


# In[28]:


cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['0', '1'], yticklabels=['0', '1'],
       xlabel='Predicted label', ylabel='True label')
ax.set_ylim(len(cm) - 0.5, -0.5)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")
fig.tight_layout()
plt.show()


# In[29]:


target_names = ['Aligned', 'Misaligned']
report = classification_report(y_test, predictions, target_names=target_names)
print(report)


# In[ ]:




