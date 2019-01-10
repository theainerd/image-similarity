# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")


import pandas as pd
import numpy as np

import gd_datagen

from keras.models import Model,Sequential
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



def get_num_classes_column_lb(column_name, df, headings_dict):

    # use for getting number of predictions for multi class classification

    lb = LabelBinarizer()
    column = df.iloc[:, headings_dict[column_name]:headings_dict[column_name]+1]
    column_np = np.array(column)
    lb.fit(column_np.astype(str))
    return (len(lb.classes_))

file_name = '../data/category_data.csv'
#getting the header
file = open(file_name, 'r')
lines = file.readlines()
file.close()
headings = lines[0].strip().split(',')
headings_dict = dict()
for i in range(len(headings)):
    headings_dict[headings[i]] = i

print(headings_dict)


experiment_name = "image-similarity"
# import data
traindf = pd.read_csv("../data/category_data.csv")
traindf['id'] = "../data/" + traindf['id']
target_labels = traindf['label']

traindf = traindf

final_model_name = experiment_name + '_inceptionv3_finetuning_final.h5'

top_layers_checkpoint_path = "../snapshots/top_layers/top_layers.h5"
fine_tuned_checkpoint_path = "../snapshots/fine_tuned/fine_tuned.h5"
new_extended_inception_weights = "../snapshots/final/final.h5"

len_df=len(traindf.id)
tr_len=int(len_df*0.8)

df_train = traindf[0:tr_len]
df_validation=traindf[tr_len:len_df]
df_overall = traindf

print('length of training dataframe (80% data)=>'+str(len(df_train.id)))
print('length of validation dataframe (80% data)=>'+str(len(df_validation.id)))

from gd_datagen import generator_from_df
ntrain = df_train.shape[0]
print('ntrain')
nvalid = df_validation.shape[0]
batch_size = 4
epochs_first = 5 # For bottleneck training
epochs_second = 30 # For Fine tuning
target_size = (224, 224)

nbatches_train, mod = divmod(ntrain, batch_size)
nbatches_valid, mod = divmod(nvalid, batch_size)

nworkers = 10

parametrization_dict = {'multi_class':[{'L1_output':'id'}],'multi_label':[]}

train_generator = generator_from_df(df_train, df_overall, headings_dict, batch_size, target_size, features=None, parametrization_dict = parametrization_dict)
validation_generator = generator_from_df(df_validation, df_overall, headings_dict, batch_size, target_size, features=None, parametrization_dict= parametrization_dict)

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
model = Model(input=base_model.input, output=predictions)

if os.path.exists(top_layers_checkpoint_path):
	model.load_weights(top_layers_checkpoint_path)
	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
##############################y code

filepath= top_layers_checkpoint_path + experiment_name + "_inceptionv3_bottleneck_{epoch:02d}_{val_acc:.2f}.h5"
##############################y code
#Save the model after every epoch.
mc_top = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=nbatches_train,
    epochs=epochs_first,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nbatches_valid,
    workers=nworkers)

model.evaluate_generator(generator=valid_generator)

if os.path.exists(fine_tuned_checkpoint_path):
	model.load_weights(fine_tuned_checkpoint_path)
	print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

filepath1 = fine_tuned_checkpoint_path + experiment_name + "_inceptionv3_bottleneck_{epoch:02d}_{val_acc:.2f}.h5"
#Save the model after every epoch.
mc_fit = ModelCheckpoint(filepath1, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


if os.path.exists(fine_tuned_checkpoint_path):
	model.load_weights(fine_tuned_checkpoint_path)
	print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=nbatches_train,
    epochs=epochs_second,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nbatches_valid,
    workers=[mc_fit])

model.evaluate_generator(generator=valid_generator)

model.save_weights(new_extended_inception_weights)
