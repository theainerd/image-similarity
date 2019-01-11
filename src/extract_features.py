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

# def get_gender_data(image_path, model):
#
#     original_width = 300
#     original_height = 300
#     final_width = 224
#     final_height = 224
#     int2class = {0:'female', 1:'male'}
#     threshold = 0.5
#     data = {"success": False}
#     response = ""
#     gender = ""
#     if image_path:
#         # read the image in PIL format
#         image = Image.open(io.BytesIO(img_response.content))
#         if image.mode != "RGB":
#             image = image.convert("RGB")
#         image = image.resize((original_width, original_height))
#         image = image.resize((final_width, final_height))
#         image = img_to_array(image)
#         image = np.expand_dims(image, axis=0)
#         image = imagenet_utils.preprocess_input(image)
#         image = np.divide(image,255.0) #rescaling
#         ans = model.predict(image, batch_size=10)
#         ans = ans[0][0] #on a scale of zero to one
#         response = ans
#
#         if (ans >= threshold):
#             data = {"success": True}
#             gender = int2class[1]
#         else:
#             data = {"success": True}
#             gender = int2class[0]
#         data['response'] = (float)(response)
#         data['gender'] = gender
#     return data



testdf = pd.read_csv("../data/category_data.csv")

model = load_model("../snapshots/fine_tuned/fine_tuned_inceptionv3_bottleneck_01_0.46.h5")

intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[
                                     model.get_layer('dense_1').output
                                     ])

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="../data/",
x_col="id",
y_col=None,
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))


test_generator.reset()
preds = intermediate_layer_model.predict_generator(test_generator,verbose=1)
print(preds)
vector = []
# for i, l in enumerate(preds):
    # print(type(l))

print(str(preds))
# print("vector: "+str(len(preds[0][0])))
