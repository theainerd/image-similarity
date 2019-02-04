from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model

print("Loading  model.\n")

final_model = load_model("../models/final_model.h5")
model_pattern = load_model("../models/finalmodel/label_pattern_inceptionv3_50_0.54.h5")

final_attribute = final_model.get_layer('mixed5').get_weights()
print(final_attribute.)

pattern_attribute = final_model.get_layer('mixed5').output
print(pattern_attribute.get_weights())