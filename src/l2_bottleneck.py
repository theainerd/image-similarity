# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD 
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import l2
import pickle
import glob
import os
from PIL import Image

from keras import backend as K
import tensorflow as tf

import os
from keras.preprocessing import image
from PIL import Image
import numpy as np
from keras.callbacks import Callback
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from sklearn.utils import class_weight
import pandas as pd


#configurations

epochs = 50
batch_size = 64
dropout = 0.5
data_dir = "../data/pattern_data_split/"
output_models_dir = "../models/L2/"
train_data_dir  = data_dir + 'train'
validation_data_dir = data_dir + 'validation'
experiment_name = "image-similarity-pattern"
img_width, img_height = 244, 244
original_img_width, original_img_height = 400, 400
final_model_name = experiment_name + '_inceptionv3_bottleneck_final.h5'
validate_images = True

traindf = pd.read_csv("../data/pattern_dataset.csv")
traindf = traindf[['_id','pattern']]
no_of_classes = len(traindf['pattern'].unique())
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(traindf['pattern']),
                                                 traindf['pattern'])

if validate_images:
    i = 0
    for filename in glob.iglob(data_dir + '**/*.*', recursive=True):
        try:
            im = Image.open(filename)
        except:
            print(filename)
            i = i + 1
            os.remove(filename)
            print(i)

confusion_matrix_directory = 'path/to/data' # format same as train

#call back for confusion matrix

class Metrics(Callback):

	def on_epoch_end(self, epoch, logs={}):
		epoch = str(epoch)
		folders = os.listdir(confusion_matrix_directory)
		folders.sort()
		confusion_matrix = [[0 for x in range(len(folders))] for y in range(len(folders))]
		matrix_file = open(experiment_name+'_'+'epoch-'+epoch+'_'+'confusion_matrix.csv','a')
		f1_file = open(experiment_name+'_f1_score.csv','a')

		for i in range(0,len(folders)):
			folder_path = confusion_matrix_directory + '/' + folders[i]
			all_images = os.listdir(folder_path)

			count = 0

			this_total = len(all_images)

			for file in all_images:
				img = image.load_img(folder_path+'/'+file, target_size=(original_img_width, original_img_height))
				img = img.resize((img_width, img_height))

				if img.mode != "RGB":
					img = image.convert("RGB")

				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = imagenet_utils.preprocess_input(x)
				x = np.divide(x,255.0)
				classes = model.predict(x, batch_size=10)
				pred = classes[0]
				top_index = 0
				for j in range(1,len(pred)):
					if (pred[top_index] < pred[j]):
						top_index = j
				confusion_matrix[i][top_index] += 1

		matrix_file.write('Void,')
		for i in range(0,len(folders)):
			matrix_file.write(folders[i])
			if (i==len(folders)-1):
				matrix_file.write('\n')
			else:
				matrix_file.write(',')

		for i in range(0,len(folders)):

			matrix_file.write(folders[i]+',')
			for j in range(len(confusion_matrix[i])):
				matrix_file.write(str(confusion_matrix[i][j]))
				if (j==len(folders)-1):
					matrix_file.write('\n')
				else:
					matrix_file.write(',')

		matrix_file.close()

		# class, precision, recall, f1-score
		f1_file.write('\n')
		f1_file.write(epoch+'\n')
		for i in range(0,len(folders)):
			f1_file.write(folders[i]+',')
			precision_denominator = 0 #number predicted to be true
			recall_denominator = 0 #number actually true
			for j in range(0,len(folders)):
				precision_denominator += confusion_matrix[j][i]
				recall_denominator += confusion_matrix[i][j]
			precision = (float(confusion_matrix[i][i])/precision_denominator)
			recall = (float(confusion_matrix[i][i])/recall_denominator)
			f1 = 2.0*precision*recall/(precision+recall)
			f1 = str(f1)
			precision = str(precision)
			recall = str(recall)
			f1_file.write(precision+','+recall+','+f1+'\n')
		f1_file.close()

metrics1 = Metrics()



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

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout)(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

predictions_color = Dense(no_of_classes, activation='softmax',name="predictions_color")(x)
predictions_pattern = Dense(no_of_classes,activation = 'softmax',name="predictions_pattern")(x)

final_output = [predictions_color,predictions_pattern]

model_not_train = Model(inputs=base_model.input, outputs=final_output)
# model_not_train.compile(optimizer = Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model_not_train.summary())


# # this is the model we will train
# model = Model(inputs=base_model.input, outputs=x)
# model.compile(optimizer = Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# filepath= output_models_dir + experiment_name + "_inceptionv3_{epoch:02d}_{val_acc:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# checkpoints =[checkpoint]
# model.fit_generator(train_generator, epochs = epochs, validation_data=validation_generator, 
# 	class_weight=class_weight, callbacks=checkpoints)
# model.save(final_model_name)
# checkpoints =[checkpoint]
# model.fit_generator(train_generator, epochs = epochs, validation_data=validation_generator, class_weight=class_weight, callbacks=checkpoints)
# model.save(final_model_name)