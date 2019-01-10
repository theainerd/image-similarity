# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")


import pandas as pd
import numpy as np
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer

import pickle
import glob
import os
from PIL import Image



experiment_name = "image-similarity"
traindf = pd.read_csv("../data/category_data.csv")
final_model_name = experiment_name + '_inceptionv3_finetuning_final.h5'

top_layers_checkpoint_path = "../snapshots/top_layers/top_layers.h5"
fine_tuned_checkpoint_path = "../snapshots/fine_tuned/fine_tuned.h5"
new_extended_inception_weights = "../snapshots/final/final.h5"

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/",
x_col="id",
y_col="label",
subset="training",
batch_size=64,
seed=42,
shuffle=True,
class_mode="sparse",
target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/",
x_col="id",
y_col="label",
subset="validation",
batch_size=1,
seed=42,
shuffle=True,
class_mode="sparse",
target_size=(224,224))


print("Downloading Base Model.....")
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 2 classes
predictions = Dense(46, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# if os.path.exists(top_layers_checkpoint_path):
# 	model.load_weights(top_layers_checkpoint_path)
# 	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

filepath= "../snapshots/top_layers/top_layers.h5"

##############################y code
#Save the model after every epoch.

mc_top = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoints =[mc_top]
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20,
                    callbacks = checkpoints)

model.evaluate_generator(generator=valid_generator)
model.save(top_layers_checkpoint_path)
