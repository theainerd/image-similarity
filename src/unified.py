# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="iEJDqOgS8QPlGv7hK3MYESLE2",
                        project_name="general", workspace="iiitian-chandan")
import gd_datagen


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Add, AveragePooling2D, Flatten, Activation
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.utils import plot_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization

import pickle
import glob
import os
from PIL import Image

from keras import backend as K
import tensorflow as tf

import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer


#---------data import



def get_num_classes_column_lb(column_name, df, headings_dict):

    # use for getting number of predictions for multi class classification

    lb = LabelBinarizer()
    column = df.iloc[:, headings_dict[column_name]:headings_dict[column_name]+1]
    column_np = np.array(column)
    lb.fit(column_np.astype(str))
    return (len(lb.classes_))

def get_num_classes_column_mlb(column_name_array, df, headings_dict):

    # use for getting number of predictions for multi label classification

    mlb =MultiLabelBinarizer()
    dummy_arr = []
    for element in column_name_array:
        dummy_arr.append(headings_dict[element])
    columns = df.iloc[:,dummy_arr[0]:dummy_arr[0]+1]
    for j in range(1, len(dummy_arr)):
        dummy_column = df.iloc[:,dummy_arr[j]:dummy_arr[j]+1]
        columns = pd.concat([columns, dummy_column], axis = 1) # stacking horizontally
    columns_np = np.array(columns)
    mlb.fit(columns_np.astype(str))
    return (len(list(mlb.classes_)))

file_name = 'balanced_data.csv'
#getting the header
file = open(file_name, 'r')
lines = file.readlines()
file.close()
headings = lines[0].strip().split(',')
headings_dict = dict()
for i in range(len(headings)):
    headings_dict[headings[i]] = i

df = pd.read_csv(file_name)
import glob
p_img=glob.glob("/home/ubuntu/hclass_ir/raw_image/*.jpg")
temp=df.path
t_i=[]
for i in range (len(temp)):
  t='/home/ubuntu/hclass_ir/raw_image/'+str(temp[i])+'.jpg'
  if t not in p_img:
    t_i.append(i)
df=df.drop(t_i)

df['path']='raw_image/'+df['path']+'.jpg'

from sklearn.utils import shuffle
df = shuffle(df)
df.reset_index(drop=True)

len_df=len(df.l1)
tr_len=int(len_df*0.8)
print('length of dataframe after filtering image=>'+str(len_df))


df_train = df[0:tr_len]
df_validation=df[tr_len:len_df]
df_overall = df
print('length of training dataframe (80% data)=>'+str(len(df_train.l1)))
print('length of validation dataframe (80% data)=>'+str(len(df_validation.l1)))


base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_output = base_model.output
base_output = Flatten()(base_output)
base_output = Dropout(0.5)(base_output)

###################################
# YASH START REVIEWING FROM HERE  #
###################################
#MLP for each levels
l1_mlp = Dense(1024)(base_output)
l1_mlp = Activation('relu')(l1_mlp)
l1_mlp = Dropout(0.5)(l1_mlp)

l2_mlp = Dense(1024, activation='relu')(base_output)
l2_mlp = Activation('relu')(l2_mlp)
l2_mlp = Dropout(0.5)(l2_mlp)

l3_mlp = Dense(1024, activation='relu')(base_output)
l3_mlp = Activation('relu')(l3_mlp)
l3_mlp = Dropout(0.5)(l3_mlp)

l4_mlp = Dense(1024, activation='relu')(base_output)
l4_mlp = Activation('relu')(l4_mlp)
l4_mlp = Dropout(0.5)(l4_mlp)


#Left Message passing block
#L1 left block
left_l1_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l1_mlp)
left_dense_L1 = Dense(1024, activation='relu')(left_l1_dense)
#L2 left block
left_l2_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l2_mlp)
left_dense_L1_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(left_dense_L1)
left_dense_l2_add = Add()([left_l2_dense, left_dense_L1_dense])
left_dense_L2 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(left_dense_l2_add)
#L3 left block
left_l3_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l3_mlp)
left_dense_L2_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(left_dense_L2)
left_dense_l3_add = Add()([left_l3_dense, left_dense_L2_dense])
left_dense_L3 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(left_dense_l3_add)
#L4 left block
left_l4_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l4_mlp)
left_dense_L3_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(left_dense_L3)
left_dense_l4_add = Add()([left_l4_dense, left_dense_L3_dense])
left_dense_L4 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(left_dense_l4_add)

# L4 right block
right_l4_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l4_mlp)
right_dense_L4 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_l4_dense)
# L3 right block
right_l3_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l3_mlp)
right_dense_L4_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_dense_L4)
right_l3_dense_add = Add()([right_l3_dense, right_dense_L4_dense])
right_dense_L3 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_l3_dense_add)
# L2 right block
right_l2_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l2_mlp)
right_dense_L3_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_dense_L3)
right_l2_dense_add = Add()([right_l2_dense, right_dense_L3_dense])
right_dense_L2 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_l2_dense_add)
# L1 right block
right_l1_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(l1_mlp)
right_dense_L2_dense = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_dense_L2)
right_l1_dense_add = Add()([right_l1_dense, right_dense_L2_dense])
right_dense_L1 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(right_l1_dense_add)


# Second message passing block
dense_L1_output_add = Add()([left_dense_L1, right_dense_L1])
dense_L1_output = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_L1_output_add)
dense_L1_output = Dropout(0.3)(dense_L1_output)

# Second message passing block
dense_L2_output_add = Add()([left_dense_L2, right_dense_L2])
dense_L2_output = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_L2_output_add)
dense_L2_output = Dropout(0.3)(dense_L2_output)

# Second message passing block
dense_L3_output_add = Add()([left_dense_L3, right_dense_L3])
dense_L3_output = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_L3_output_add)
dense_L3_output = Dropout(0.3)(dense_L3_output)

# Second message passing block
dense_L4_output_add = Add()([left_dense_L4, right_dense_L4])
dense_L4_output = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(dense_L4_output_add)
dense_L4_output = Dropout(0.3)(dense_L4_output)



#category_predictions = Dense(get_num_classes_column_lb('l1', df_overall, headings_dict), activation='softmax', name  = 'category_output')(dense_category_output)
#sub_category_predictions = Dense(get_num_classes_column_lb('l2', df_overall, headings_dict), activation='softmax', name = 'sub_category_output')(dense_sub_category_output)


L1_output_predictions = Dense(get_num_classes_column_lb('l1', df_overall, headings_dict), activation='softmax', name  = 'L1_output')(dense_L1_output)
L2_output_predictions = Dense(get_num_classes_column_lb('l2', df_overall, headings_dict), name  = 'L2_output')(dense_L2_output)
L3_output_predictions = Dense(get_num_classes_column_lb('l3', df_overall, headings_dict), name  = 'L3_output')(dense_L3_output)
L4_output_predictions = Dense(get_num_classes_column_lb('l4', df_overall, headings_dict), activation='softmax', name  = 'L4_output')(dense_L4_output)

model = Model(input=base_model.input, outputs=[L1_output_predictions,L2_output_predictions,L3_output_predictions,L4_output_predictions])

model.compile(optimizer='rmsprop', metrics=['accuracy'], loss={'L1_output': 'categorical_crossentropy', 'L2_output' : 'categorical_crossentropy', 'L3_output' : 'categorical_crossentropy', 'L4_output' : 'categorical_crossentropy'})

from gd_datagen import generator_from_df
ntrain = df_train.shape[0]
print('ntrain')
nvalid = df_validation.shape[0]
batch_size = 4
epochs = 25
target_size = (224, 224)

parametrization_dict = {'multi_class':[{'L1_output':'l1'}],'multi_label':[]}

train_generator = generator_from_df(df_train, df_overall, headings_dict, batch_size, target_size, features=None, parametrization_dict = parametrization_dict)
validation_generator = generator_from_df(df_validation, df_overall, headings_dict, batch_size, target_size, features=None, parametrization_dict= parametrization_dict)

nbatches_train, mod = divmod(ntrain, batch_size)
nbatches_valid, mod = divmod(nvalid, batch_size)

nworkers = 10

# checkpoint
filepath="weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=nbatches_train,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nbatches_valid,
    workers=nworkers)

model.save('model.h5')
