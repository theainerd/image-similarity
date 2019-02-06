# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oWiH86Pi5sqYSaVZmV1BYxBls",
                        project_name="image-similarity", workspace="theainerd")

# from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD 
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
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
no_of_classes = 6
epochs = 50
batch_size = 64
dropout = 0.5
data_dir = "../data/sleeve_len_balanced/"
output_models_dir = "../models/label_sleevelen_bottleneck/"
train_data_dir  = data_dir + 'train'
validation_data_dir = data_dir + 'validation'
experiment_name = "label_pattern_Resnet50"
img_width, img_height = 299,299
original_img_width, original_img_height = 400, 400
final_model_name = experiment_name + '_bottleneck.h5'
validate_images = True

traindf = pd.read_csv("../data/pattern_balanced.csv")
traindf = traindf[['_id','pattern']]
traindf = traindf[traindf.pattern != "geometric"]
traindf = traindf[traindf.pattern != "fotoprint"]
traindf = traindf[traindf.pattern != "paisley"]
traindf = traindf[traindf.pattern != "stud"]
traindf = traindf[traindf.pattern != "rivets"]
traindf = traindf[traindf.pattern != "pinstripe"]
traindf = traindf[traindf.pattern != "flounce"]
traindf = traindf[traindf.pattern != "gemstones"]


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

class_weight = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)

print(class_weight)

print("Downloading Base Model.....")

base_model = ResNet50(include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

# get layers and add average pooling layer
## set model architechture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(dropout)(x)
predictions = Dense(no_of_classes, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)
model.compile(optimizer=Adam(0.001), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', 'accuracy'])

filepath= output_models_dir + experiment_name + "_inceptionv3_{epoch:02d}_{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoints =[checkpoint]
model.fit_generator(train_generator, epochs = epochs,steps_per_epoch=278,validation_steps = 70, validation_data=validation_generator,class_weight = class_weight, callbacks=checkpoints)
model.save(final_model_name)