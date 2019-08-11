
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[29]:


height = pd.read_csv('height_train.csv')
test = pd.read_csv('height_test.csv')
real = pd.read_csv('height_real.csv')


# In[15]:


model = LinearRegression()
model.fit(X=height.loc[:,['father_height','mother_height','boy_dummy']],y=height.child_height)


# In[16]:


model.coef_


# In[17]:


model.intercept_


# In[22]:


# 1. 构造方程预测
test['prediction'] = 0.23959427*test.father_height+0.25013358*test.mother_height+0.10030806*test.boy_dummy+0.8274299645517075


# In[32]:


# 2. 直接使用模型的predict()
test['prediction'] = model.predict(test.loc[:,['father_height','mother_height','boy_dummy']])


# In[ ]:


def evaluate(prediction_path,real_path):
    predict = pd.read_csv(prediction_path)
    real = pd.read_csv(real_path)
    predict = predict.loc[:,['id','prediction']]
    real = real.loc[:,['id','child_height']]
    real = real.merge(predict,on='id',how='left')
    return np.square(real.prediction*100-real.child_height*100).mean()


# In[34]:


np.square(test.prediction*100-real.child_height*100).mean()


# In[30]:


real

