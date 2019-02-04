from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras import layers

print("Loading  model.\n")

final_model = load_model("../models/final_model.h5")
model_pattern = load_model("../models/finalmodel/label_pattern_inceptionv3_50_0.54.h5")

# for i,layer in enumerate(final_model.layers):
#     print(i,layer.name)


for i,layer in enumerate(model_pattern.layers):
    print(i,layer.name)

# 
# for i,layer in enumerate(final_model.layers[289:291]):
#     weights = layer.get_weights()
#     print(i,weights)

# for i,layer1 in enumerate(model_pattern.layers[289:291]):
#     weights1 = layer1.get_weights()
#     print(i,weights1)
