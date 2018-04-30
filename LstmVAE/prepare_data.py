# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:45:14 2018

@author: Dany
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess(df):
    
    """returns normalized and standardized data.
    """
    
    df = np.asarray(df, dtype=np.float32)
    
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')
        
    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()
    
    #standardize data 
    df = StandardScaler().fit_transform(df)
    print('Data standardized')
    #normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')
    
    return df
     
