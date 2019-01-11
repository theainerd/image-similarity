import pandas as pd
import numpy as np
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,GlobalMaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import os

from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer

testdf = pd.read_csv("../data/category_data.csv")
testdf = testdf[:4]

model = load_model("../snapshots/fine_tuned/fine_tuned_inceptionv3_bottleneck_01_0.46.h5")

intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[
                                     model.get_layer('dense_1').output
                                     ])

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="./data/",
x_col="id",
y_col=None,
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))

test_generator.reset()
preds = model.predict_generator(test_generator,verbose=1)
vector = []
for i, l in enumerate(preds):
    print(type(l))

print(str(preds))
print("vector: "+str(len(preds[0][0])))
