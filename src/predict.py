from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from keras.models import load_model

final_model = load_model("../models/final_model.h5")
final_width = 229
final_height = 229
image = Image.open("img_00000005.jpg")
image = image.resize((final_width, final_height))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# image = imagenet_utils.preprocess_input(image)
image = np.divide(image,255.0) #rescaling
ans = final_model.predict(image, batch_size=10)
print(ans)
