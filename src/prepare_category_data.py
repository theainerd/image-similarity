import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.utils import shuffle

traindf = pd.read_csv("../data/category_data.csv")
list1 = set(traindf['label'].tolist())
print(list1)