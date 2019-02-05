from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras import layers

print("Loading  model.\n")

model = load_model("../models/model_12_128.h5")
# model_pattern = load_model("../models/finalmodel/label_pattern_inceptionv3_50_0.54.h5")

from keras.utils import plot_model
plot_model(model, to_file='model_yash.png')