import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.utils import shuffle

traindf = pd.read_csv("../data/category_data.csv")
shuffle(traindf)
print(traindf.shape)
print(traindf['label'].value_counts())
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(traindf['label']),
                                                 traindf['label'])

list1 = list(np.unique(traindf['label']))
list2 = list(class_weights)

for i,j in zip(list1,list2):
	print (i,j)