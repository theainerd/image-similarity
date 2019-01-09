import pandas as pd
import numpy as np
from keras.models import Model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

# # import comet_ml in the top of your file
# from comet_ml import Experiment
#
# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="",
#                         project_name="fashion-object-detection")
#
experiment_name = "image-similarity"
# import data
traindf = pd.read_csv("../data/category_data.csv")
traindf['id'] = "../data/" + traindf['id']

target_labels = traindf['label']


labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
labels_ohe = np.asarray(labels_ohe_names)
print(labels_ohe.shape)

train_data = np.array([img_to_array(load_img(img,target_size=(299, 299))
                       ) for img in traindf['id'].values.tolist()]).astype('float32')

print("Train data shape is: " + train_data.shape)

x_train, x_val, y_train, y_val = train_test_split(train_data,
                                                    target_labels,
                                                    test_size=0.15,
                                                    stratify=np.array(y_train),
                                                    random_state=42)

y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()

BATCH_SIZE = 32


final_model_name = experiment_name + '_inceptionv3_finetuning_final.h5'

top_layers_checkpoint_path = "../snapshots/top_layers"
fine_tuned_checkpoint_path = "../snapshots/fine_tuned"
new_extended_inception_weights = "../snapshots/final"

datagen=ImageDataGenerator(rescale=1./255.,
        shear_range=0.2,
        zoom_range=0.5,
        width_shift_range=0.5,
        horizontal_flip =True)


# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = 'true')

train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False, batch_size=BATCH_SIZE, seed=1)

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False, batch_size=BATCH_SIZE, seed=1)

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
                    epochs=20
)

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

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20,
                    callbacks=[mc_fit, tb]
)

model.evaluate_generator(generator=valid_generator)

model.save_weights(new_extended_inception_weights)
