import pandas as pd
import numpy as np

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

experiment_name = "image-similarity"
# import data
traindf = pd.read_csv("../data/category_data.csv")
output_models_dir = "../snapshots/top_layers"
final_model_name = experiment_name + '_inceptionv3_finetuning_final.h5'

top_layers_checkpoint_path = "../snapshots/top_layers"
fine_tuned_checkpoint_path = "../snapshots/fine_tuned"
new_extended_inception_weights = "../snapshots/final"

datagen=ImageDataGenerator(rescale=1./255.,
        shear_range=0.2,
        zoom_range=0.5,
        width_shift_range=0.5,
        horizontal_flip=True,validation_split=0.10)

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/img/",
x_col="id",
y_col="label",
subset="training",
batch_size=2,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))


valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/img/",
x_col="id",
y_col="label",
subset="validation",
batch_size=2,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))

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

#Save the TensorBoard logs.
tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

model.evaluate_generator(generator=valid_generator)
model.save
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

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    callbacks=[mc_fit, tb]
)

model.evaluate_generator(generator=valid_generator)

model.save_weights(new_extended_inception_weights)
