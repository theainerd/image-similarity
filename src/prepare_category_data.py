import pandas as pd
import numpy as np

traindf = pd.read_csv("../data/category_data.csv")
traindf['vector'] = 2
traindf['id'] = "../data/"+traindf['id']
traindf.to_csv("../data/category_data_new.csv",index=False)
