import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


test_set = pd.read_csv('raw/TestSet.csv')
train_set = pd.read_csv('raw/TrainingSet.csv')
test_subset = pd.read_csv('raw/TestSubset.csv')
train_subset = pd.read_csv('raw/TrainingSubset.csv')

train = train_set.drop(['EbayID','QuantitySold','SellerName'],axis=1)
train_target = train_set['QuantitySold']

_,n_features = train.shape

df = DataFrame(np.hstack((train,train_target[:,None])),columns=range(n_features)+["isSold"])
_=sns.pairplot(df[:50],vars=[2,3,4,10,13],hue="isSold",size=1.5)
