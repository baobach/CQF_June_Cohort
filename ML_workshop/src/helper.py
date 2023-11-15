# Machine Learning Workshop - I
# Kannan Singaravelu

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin


# define seed
def set_seeds(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

# create custom day transformer 
class DayTransformer(BaseEstimator, TransformerMixin):
                                   
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.data = pd.DataFrame(
            {
        'WeekDay': ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            }
        )
        self.daysnum = np.array(self.data.index+1)
        return self
    
    
    def transform(self, X): # X is a dataframe
        Xt = X.copy()
        pi = np.pi
        num = Xt.index.weekday+1
        
        Xt['dsin'] = np.sin(2 * pi * num / np.max(self.daysnum))
        Xt['dcos'] = np.cos(2 * pi * num / np.max(self.daysnum))
        Xt = Xt.drop(['days'], axis=1)
        
        return Xt

    
# create custom time transformer 
class TimeTransformer(BaseEstimator, TransformerMixin):
                                   
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.data = pd.DataFrame(
            {
        'DayParts': ["afternoon","morning","noon"]
            }
        )
        self.timenum = np.array(self.data.index+1)
        return self
    
    
    def transform(self, X):
        Xt = X.copy()
        pi = np.pi
        num = Xt.hours.apply(lambda x: 1 if x=='afternoon' else (2 if x=='morning' else 3))
        Xt['tsin'] = np.sin(2 * pi * num / np.max(self.timenum))
        Xt['tcos'] = np.cos(2 * pi * num / np.max(self.timenum))
        Xt = Xt.drop(['hours'], axis=1)
        
        return Xt


# create function to read locally stored file
def getdata(filename):
    df = pd.read_csv('./data/'+filename+'.csv')
    df.datetime = pd.to_datetime(df.datetime)
    df = (
        df.set_index('datetime', drop=True)
        .drop('symbol', axis=1)
    )
    
    # add days
    df['days'] = df.index.day_name()

    # add dayparts
    df['hours'] = df.index.hour
    df['hours'] = df['hours'].apply(daypart)

    return df


# create function to group trade hours
def daypart(hour):
    if hour in [9,10,11]:
        return "morning"
    elif hour in [12,13]:
        return "noon"
    elif hour in [14,15,16,17,18,19]:
        return "afternoon"


# class weight function
def cwts(dfs):
    c0, c1 = np.bincount(dfs)
    w0=(1/c0)*(len(dfs))/2 
    w1=(1/c1)*(len(dfs))/2 
    return {0: w0, 1: w1}

# Machine Learning Workshop - I by Kannan Singaravelu
# November 2021
