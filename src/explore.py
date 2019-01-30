import pandas as pd 
import numpy as np 

traindf = pd.read_csv("../data/pattern_balanced.csv")
print(traindf['pattern'].value_counts())
url = traindf[traindf.pattern == "dots"]['media.standard.0.url'].tolist()
print(url[180:190])