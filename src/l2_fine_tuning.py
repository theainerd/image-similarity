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

from keras.callbacks import *
from clr_callback import *

from sklearn.utils import class_weight
import pandas as pd


#configurations
epochs = 50
batch_size = 64
dropout = 0.5
no_of_classes = 46
data_dir = "../data/pattern_data_split//"
# base_model_path = "models/L2/IntuL2-classification_inceptionv3_bottleneck_16_0.61.h5"
base_model_path = "../models/L2/image-similarity-pattern_inceptionv3_03_0.54.h5"
output_models_dir = "../models/L2-fine/"
train_data_dir  = data_dir + 'train'
validation_data_dir = data_dir + 'validation'
experiment_name = "image-similarity-finetuning-pattern"

img_width, img_height = 299, 299
final_model_name = experiment_name + '_inception_pattern_finetuning_final.h5'

confusion_matrix_directory = 'path/to/data' # format same as train
original_img_width, original_img_height = 400, 400

traindf = pd.read_csv("../data/pattern_dataset.csv")
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

model = load_model(base_model_path)

for layer in model.layers[:172]:
  layer.trainable = False
for layer in model.layers[172:]:
  layer.trainable = True

clr_triangular = CyclicLR(mode='triangular')
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', 'accuracy'])

filepath= output_models_dir + experiment_name + "_inceptionv3_finetuning_{epoch:02d}_{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoints =[checkpoint,clr_triangular]
model.fit_generator(train_generator, epochs = epochs, validation_data=validation_generator, class_weight=class_weight, callbacks=checkpoints)
model.save(final_model_name)