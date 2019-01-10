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

from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer


model = load_model("../snapshots/top_layers/top_layers.h5")
experiment_name = "image-similarity"
traindf = pd.read_csv("../data/category_data.csv")
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
batch_size=32,
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


if os.path.exists(fine_tuned_checkpoint_path):
	model.load_weights(fine_tuned_checkpoint_path)
	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")


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

filepath = fine_tuned_checkpoint_path
#Save the model after every epoch.
mc_fit = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


model.fit_generator(
    generator=train_generator,
    steps_per_epoch=nbatches_train,
    epochs=epochs_second,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nbatches_valid,
    workers=[mc_fit])

model.save(new_extended_inception_weights)
