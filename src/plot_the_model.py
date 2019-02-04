from keras.models import load_model,
from keras.models import Model
from keras.utils import plot_model

print("Loading  model.\n")

final_model = load_model("../models/final_model.h5")
model_pattern = load_model("../models/finalmodel/label_pattern_inceptionv3_50_0.54.h5")

final_attribute = final_model.get_layer('mixed5').output
final_attribute = model_pattern.get_layer('conv2d_55')(final_attribute)

for i,layer in enumerate(final_model.layers):
    weights = layer.get_weights()
    print(i,weights)

pattern_attribute = final_model.get_layer('mixed5').output
print(pattern_attribute.get_weights())