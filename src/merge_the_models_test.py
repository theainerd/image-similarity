from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model

print("Loading color model.\n")
model_color = load_model("../models/finalmodel/label_color_inceptionv3_49_0.35.h5")

print("Preparing Color Model...")

for layer in model_color.layers:
    layer.name = layer.name + str("_xyz")

print(model_color.summary())