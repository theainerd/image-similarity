# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l2
import pickle

from keras import backend as K
import tensorflow as tf

import os
from keras.preprocessing import image
from PIL import Image
import numpy as np
from keras.callbacks import Callback
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

# from keras.callbacks import *
# from clr_callback import *

from sklearn.utils import class_weight
import pandas as pd


#configurations
epochs = 50
batch_size = 64
dropout = 0.5
# no_of_classes = 46
data_dir = "../data/color_balanced/"
# base_model_path = "models/L2/IntuL2-classification_inceptionv3_bottleneck_16_0.61.h5"
# base_model_path = "../models/L2/label_pattern_inceptionv3_10_0.15.h5"
output_models_dir = "../models/label_color/"
train_data_dir  = data_dir + 'train'
validation_data_dir = data_dir + 'validation'
experiment_name = "multiclass"

img_width, img_height = 299, 299
final_model_name = experiment_name + '_inception_pattern_finetuning_final.h5'

confusion_matrix_directory = 'path/to/data' # format same as train
original_img_width, original_img_height = 400, 400

traindf = pd.read_csv("../data/color_balanced.csv")
traindf = traindf[['_id','color']]
no_of_classes = len(traindf['color'].unique())
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(traindf['color']),
                                                 traindf['color'])


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

base_model = InceptionV3(weights='imagenet', include_top=False)
                          
# pattern_attribute = output
# pattern_attribute = model.get_layer('global_average_pooling2d_1')(pattern_attribute)
# pattern_attribute = model.get_layer('dropout_1')(pattern_attribute)
# pattern_attribute = model.get_layer('attribute_pattern')(pattern_attribute)
# predictions_pattern = model.get_layer('predictions_pattern')(pattern_attribute)

color_attribute = base_model.output
color_attribute = GlobalAveragePooling2D(name='global_average_pooling2d_2')(color_attribute)
color_attribute = Dropout(dropout,name='dropout_2')(color_attribute)
color_attribute = Dense(1024, activation='relu',name = "attribute_color")(color_attribute)
predictions_color = Dense(17, activation='softmax',name="predictions_color")(color_attribute)

model = Model(inputs=base_model.input, outputs = predictions_color)

# change this code for every attribute - set the layers to true for training
for layer in base_model.layers:
    layer.trainable = False

# model1 = Model(inputs = model.input, outputs = predictions_color)

# for layer in model1.layers:
#   layer.trainable = False

# for layer in model1.layers[305:]:
#   print(layer.name)
#   layer.trainable = False


# from keras.utils import plot_model
# plot_model(model1, to_file='model1.png')

model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

filepath= output_models_dir + experiment_name + "multiclass_{epoch:02d}_{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoints =[checkpoint]
model.fit_generator(train_generator, epochs = epochs,class_weight = class_weight, validation_data=validation_generator, callbacks=checkpoints)
model.save(final_model_name)