from keras.models import load_model
from keras.models import Model

model_pattern = load_model("../models/label_pattern_bottleneck/label_pattern_inceptionv3_03_0.39.h5")
model_color = load_model("../models/label_color_bottleneck/label_color_inceptionv3_05_0.22.h5")

for layer in model_color.layers:
    layer.name = layer.name + str("_color")


pattern_attribute = model_pattern.get_layer('mixed10').output
pattern_attribute = model_pattern.get_layer('global_average_pooling2d_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('dropout_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('dense_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('dropout_2')(pattern_attribute)
predictions_pattern = model_pattern.get_layer('dense_2')(pattern_attribute)

print(model_color.summary())
color_attribute = model_pattern.get_layer('mixed10').output
color_attribute = model_color.get_layer('global_average_pooling2d_1_color')(color_attribute)
color_attribute = model_color.get_layer('dropout_1_color')(color_attribute)
color_attribute = model_color.get_layer('attribute_color_color')(color_attribute)
color_attribute = model_color.get_layer('dropout_2_color')(color_attribute)
predictions_color = model_color.get_layer('predictions_color_color')(color_attribute)

final_model = Model(inputs= model_pattern.input, outputs= [predictions_pattern,predictions_color])


final_model.save("../models/final_model_new.h5")
