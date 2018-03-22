# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:46:15 2018

@author: 刘欢
"""
import numpy as np
class NearestNeighbor:
    def _init_(self):
        pass
    def train(self,X,y):
        """X is N X D where each row is an example.y is 1-dimension of size N"""
        self.Xtr=X
        self.ytr=y
    def predict(self,X):
        num_test=X.shape[0]#obtain the 0 dimension length of the X
        Ypred=np.zeros(num_test,dtype=self.ytr.dtype)
        for i in xrange(num_test):
            #using the L1 distance 
            distances=np.sum(np.abs(self.Xtr-X[i,:],axis=1))
            min_index=np.argmin(distances)
            #predict the label of the nearest example
            Ypred[i]=self.ytr[min_index]
        return Ypred
        
                                    
