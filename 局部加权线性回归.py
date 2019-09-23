import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class LWLR(object):
    
    def __init__(self,k=1): #Smaller k means more focusing weights.
        self.k = k
        
    def fit(self,train,target):#All inputs should be numpy arrays.
        self.train = train
        self.target = target
        return self
    
    def predict_single(self,example):
        weights = np.square(self.train-example)
        weights = np.sum(weights,axis=1)
        weights = -0.5*weights/np.square(self.k)
        weights = np.exp(weights)
        model = LinearRegression()
        model.fit(X=self.train,y=self.target,sample_weight=weights)
        return model.predict(np.reshape(example,[1,-1]))[0]
        
        
    def predict(self,test):#Return predictions as a numpy array.
        result = []
        for example in test:
            result.append(self.predict_single(example))
        return np.array(result)
    

# 在孩子身高数据集上使用
train = pd.read_csv('height_train.csv')
test = pd.read_csv('height_test.csv')
real = pd.read_csv('height_real.csv')

lwlr = LWLR(k=0.2)
lwlr.fit(train.loc[:,['father_height','mother_height','boy_dummy']].values,train.child_height.values)
test['prediction']=lwlr.predict(test.loc[:,['father_height','mother_height','boy_dummy']].values)

print(np.square(test.prediction*100-real.child_height*100).mean())
