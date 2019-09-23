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