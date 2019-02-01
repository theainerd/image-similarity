from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model

model_pattern = load_model("../models/finalmodel/label_color_inceptionv3_49_0.35.h5")
# plot_model(model_pattern, to_file='model_pattern.png')
print(model_pattern.summary())