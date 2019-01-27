# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")

# from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD 
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import l2
import pickle
import glob
import os
from PIL import Image

from keras import backend as K
import tensorflow as tf

import os
from keras.preprocessing import image
from PIL import Image
import numpy as np
from keras.callbacks import Callback
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from sklearn.utils import class_weight
import pandas as pd


#configurations

epochs = 50
batch_size = 64
dropout = 0.5
data_dir = "../data/pattern_balanced_split/"
output_models_dir = "../models/label_pattern/"
train_data_dir  = data_dir + 'train'
validation_data_dir = data_dir + 'validation'
experiment_name = "label_pattern"
img_width, img_height = 244, 244
original_img_width, original_img_height = 400, 400
final_model_name = experiment_name + '_inceptionv3_bottleneck_final.h5'
validate_images = True

traindf = pd.read_csv("../data/pattern_balanced.csv")
traindf = traindf[['_id','pattern']]
no_of_classes = len(traindf['pattern'].unique())
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(traindf['pattern']),
                                                 traindf['pattern'])

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode="categorical",
	shuffle=True)

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode="categorical",
	shuffle=True)

print("Downloading Base Model.....")

base_model = VGG16(weights='imagenet', include_top=False)

# pattern attribute layer

pattern_attribute = base_model.output
pattern_attribute = GlobalAveragePooling2D(name = 'global_average_pooling2d_1')(pattern_attribute)
pattern_attribute = Dropout(dropout)(pattern_attribute)
# let's add a fully-connected layer
pattern_attribute_layer = Dense(1024, activation='relu',name = "attribute_pattern")(pattern_attribute)
predictions_pattern = Dense(no_of_classes,activation = 'softmax',name="predictions_pattern")(pattern_attribute_layer)

model = Model(inputs=base_model.input, outputs = predictions_pattern)


# model.load_weights("../models/label_pattern/label_pattern_inceptionv3_41_0.37.h5")
# print ("Checkpoint loaded.")

# change this code for every attribute - set the layers to true for training
for layer in base_model.layers:
    layer.trainable = False

# this is the model we will train

model.compile(optimizer = SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

filepath= output_models_dir + experiment_name + "_inceptionv3_{epoch:02d}_{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoints =[checkpoint]
model.fit_generator(train_generator, epochs = epochs, validation_data=validation_generator,class_weight = class_weight, callbacks=checkpoints)
model.save(final_model_name)