from keras.models import load_model
from keras.models import Model

model_pattern = load_model("../models/label_pattern/label_pattern_inceptionv3_10_0.15.h5")

for i,layer in enumerate(model_pattern.layers):
    print(i,layer.name)

pattern_attribute = model_pattern.get_layer('mixed5').output
pattern_attribute = model_pattern.get_layer('mixed5')(pattern_attribute)
print(pattern_attribute)


# pattern_attribute = model_pattern.get_layer('global_average_pooling2d_1')(pattern_attribute)
# pattern_attribute = model_pattern.get_layer('dropout_1')(pattern_attribute)
# pattern_attribute = model_pattern.get_layer('attribute_pattern')(pattern_attribute)
# predictions_pattern = model_pattern.get_layer('predictions_pattern')(pattern_attribute)
