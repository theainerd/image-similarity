# # import comet_ml in the top of your file
# from comet_ml import Experiment
#
# # Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
#                         project_name="image-similarity", workspace="theainerd")
#

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
from keras.optimizers import Adam

from keras.callbacks import *
from clr_callback import *

import pandas as pd
import numpy as np
import os

from sklearn.utils import class_weight
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer

from sklearn.utils import shuffle

experiment_name = "image-similarity"

traindf = pd.read_csv("../data/category_data.csv")
traindf = traindf[['id','label']]
traindf = shuffle(traindf)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(traindf['label']),
                                                 traindf['label'])


top_layers_checkpoint_path = "../snapshots/top_layers/"
fine_tuned_checkpoint_path = "../snapshots/fine_tuned/fine_tuned.h5"
new_extended_inception_weights = "../snapshots/final/final.h5"

datagen=ImageDataGenerator(rescale=1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/",
x_col="id",
y_col="label",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/",
x_col="id",
y_col="label",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))


print("Downloading Base Model.....")
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalMaxPooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 2 classes
predictions = Dense(46, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

if os.path.exists(top_layers_checkpoint_path):
	model.load_weights(top_layers_checkpoint_path)
	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")


# if os.path.exists(top_layers_checkpoint_path):
# 	model.load_weights(top_layers_checkpoint_path)
# 	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
clr_triangular = CyclicLR(mode='triangular')
model.compile(optimizer= Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

##############################y code
#Save the model after every epoch.
filepath= top_layers_checkpoint_path + "toplayer_inceptionv3_bottleneck_{epoch:02d}_{val_acc:.2f}.h5"
mc_top = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20,
                    class_weight=class_weights,
                    callbacks = [mc_top,clr_triangular])


model.evaluate_generator(generator=valid_generator)
model.save(top_layers_checkpoint_path)
