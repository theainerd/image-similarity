import pandas as pd
import numpy as np

from PIL import Image
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

model = load_model("../snapshots/fine_tuned/fine_tuned_inceptionv3_bottleneck_03_0.58.h5")
# model = load_model("../snapshots/fine_tuned/fine_tuned_inceptionv3_bottleneck_01_0.46.h5")

intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[
                                     model.get_layer('dense_1').output
                                     ])

def extract_vector(image_path):
    final_width = 229
    final_height = 229
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((final_width, final_height))
    image = img_to_array(image)
    image = np.divide(image,255.0) #rescaling
    image = np.expand_dims(image, axis=0)
    preds = intermediate_layer_model.predict(image)
    preds = preds[0]
    preds = preds.tolist()
    print("Extracting Image"+image_path)
    return preds



traindf = pd.read_csv("../data/category_data.csv")
traindf['id'] = "../data/"+traindf['id']
traindf['vectors'] = traindf['label']
traindf = traindf[:2]
traindf['vectors'] = traindf['id'].map(lambda x: extract_vector(x))
traindf = traindf['vectors']

traindf.to_pickle("../data/feature_vector.pkl")
