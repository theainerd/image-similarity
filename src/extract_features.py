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

image = Image.open(io.BytesIO(img_response.content))
if image.mode != "RGB":
    image = image.convert("RGB")
image = image.resize((original_width, original_height))
image = image.resize((final_width, final_height))
image = img_to_array(image)
image = imagenet_utils.preprocess_input(image)
image = np.divide(image,255.0) #rescaling
image = np.expand_dims(image, axis=0)

testdf = pd.read_csv("../data/category_data.csv")

model = load_model("../snapshots/fine_tuned/fine_tuned_inceptionv3_bottleneck_01_0.46.h5")

intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[
                                     model.get_layer('dense_1').output
                                     ])

preds = intermediate_layer_model.predict(image)
print(preds)
vector = []
# for i, l in enumerate(preds):
    # print(type(l))

print(str(preds))
# print("vector: "+str(len(preds[0][0])))
