from keras.models import load_model
from keras.models import Model

model_pattern = load_model("../models/label_pattern/label_pattern_inceptionv3_10_0.15.h5")
model_color = load_model("../models/label_pattern/multiclass_02_0.10.h5")

pattern_attribute = model_pattern.get_layer('mixed10').output
pattern_attribute = model_pattern.get_layer('global_average_pooling2d_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('dropout_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('attribute_pattern')(pattern_attribute)
predictions_pattern = model_pattern.get_layer('predictions_pattern')(pattern_attribute)

# color_attribute = model.get_layer('global_average_pooling2d_2')(color_attribute)
# color_attribute = model.get_layer('dropout_2')(color_attribute)
# color_attribute = model.get_layer('attribute_color')(color_attribute)
# color_attribute = model.get_layer('predictions_color')(color_attribute)

color_attribute = model_pattern.get_layer('mixed10').output
color_attribute = model_color.get_layer('global_average_pooling2d_2')(color_attribute)
color_attribute = model_color.get_layer('dropout_2')(color_attribute)
color_attribute = model_color.get_layer('attribute_color')(color_attribute)
predictions_color = model_color.get_layer('predictions_color')(color_attribute)

final_model = Model(inputs= model_pattern.input, outputs= [predictions_pattern,predictions_color])

print(final_model.summary())
final_model.save("../models/final_model.h5")
