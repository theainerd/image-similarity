# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")


import pandas as pd
import numpy as np
import os

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

from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer

from sklearn.utils import shuffle



model = load_model("../snapshots/top_layers/top_layers.h5")
experiment_name = "image-similarity"

traindf = pd.read_csv("../data/category_data.csv")
traindf = traindf[['id','label']]
traindf = shuffle(traindf)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(traindf['label']),
                                                 traindf['label'])

traindf = shuffle(traindf)

top_layers_checkpoint_path = "../snapshots/top_layers/top_layers.h5"
fine_tuned_checkpoint_path = "../snapshots/fine_tuned/"
new_extended_inception_weights = "../snapshots/final/final.h5"


datagen=ImageDataGenerator(rescale=1./255.,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.20)

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


if os.path.exists(fine_tuned_checkpoint_path):
	model.load_weights("../snapshots/fine_tuned/fine_tuned_inceptionv3_bottleneck_03_0.55.h5")
	print ("Checkpoint" + fine_tuned_checkpoint_path + " loaded.")
#

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import Adam
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)

filepath= fine_tuned_checkpoint_path + "fine_tuned_inceptionv3_bottleneck_{epoch:02d}_{val_acc:.2f}.h5"
#Save the model after every epoch.
mc_fit = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20,
                    class_weight=class_weights
                    callbacks = [clr_triangular,mc_fit])



model.evaluate_generator(generator=valid_generator)
model.save(new_extended_inception_weights)
