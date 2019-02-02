from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model

print("Loading pattern model.\n")
model_pattern = load_model("../models/finalmodel/label_pattern_inceptionv3_50_0.54.h5")
print("Loading color model.\n")
model_color = load_model("../models/finalmodel/label_color_inceptionv3_49_0.35.h5")
print("Loading gender model.\n")
model_gender = load_model("../models/finalmodel/label_gender_inceptionv3_08_0.83.h5")
print("Preparing Pattern Model.")

for layer in model_color.layers:
    layer.name = layer.name + str("_color")

for layer in model_gender.layers:
    layer.name = layer.name + str("_gender")


##########################################################################################################################
################################################## Model for Pattern #####################################################
##########################################################################################################################

pattern_attribute = model_pattern.get_layer('mixed5').output
pattern_attribute = model_pattern.get_layer('conv2d_55')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_55')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_55')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_56')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_56')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_56')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_57')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_57')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_57')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_58')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_58')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_58')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_59')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_59')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_59')(pattern_attribute)


pattern_attribute_branch = model_pattern.get_layer('mixed5').output
pattern_attribute_branch = model_pattern.get_layer('conv2d_52')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_52')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_52')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('conv2d_53')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_53')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_53')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('conv2d_54')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_54')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_54')(pattern_attribute_branch)


pattern_attribute_branch_two = model_pattern.get_layer('mixed5').output
pattern_attribute_branch_two = model_pattern.get_layer('average_pooling2d_6')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('conv2d_60')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('batch_normalization_60')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('activation_60')(pattern_attribute_branch_two)

pattern_attribute_branch_three = model_pattern.get_layer('mixed5').output
pattern_attribute_branch_three = model_pattern.get_layer('conv2d_51')(pattern_attribute_branch_three)
pattern_attribute_branch_three = model_pattern.get_layer('batch_normalization_51')(pattern_attribute_branch_three)
pattern_attribute_branch_three = model_pattern.get_layer('activation_51')(pattern_attribute_branch_three)

# Merge layers
x = [pattern_attribute,pattern_attribute_branch,pattern_attribute_branch_two,pattern_attribute_branch_three]

merge_pattern_six = model_pattern.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


pattern_attribute = model_pattern.get_layer('conv2d_65')(merge_pattern_six)
pattern_attribute = model_pattern.get_layer('batch_normalization_65')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_65')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_66')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_66')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_66')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_67')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_67')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_67')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_68')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_68')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_68')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_69')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_69')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_69')(pattern_attribute)


pattern_attribute_branch = model_pattern.get_layer('conv2d_62')(merge_pattern_six)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_62')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_62')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('conv2d_63')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_63')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_63')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('conv2d_64')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_64')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_64')(pattern_attribute_branch)

pattern_attribute_branch_two = model_pattern.get_layer('average_pooling2d_7')(merge_pattern_six)
pattern_attribute_branch_two = model_pattern.get_layer('conv2d_70')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('batch_normalization_70')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('activation_70')(pattern_attribute_branch_two)

pattern_attribute_branch_three = model_pattern.get_layer('conv2d_61')(merge_pattern_six)
pattern_attribute_branch_three = model_pattern.get_layer('batch_normalization_61')(pattern_attribute_branch_three)
pattern_attribute_branch_three = model_pattern.get_layer('activation_61')(pattern_attribute_branch_three)

# Merge layers
x = [pattern_attribute,pattern_attribute_branch,pattern_attribute_branch_two,pattern_attribute_branch_three]

merge_pattern_seven = model_pattern.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

pattern_attribute = model_pattern.get_layer('conv2d_73')(merge_pattern_seven)
pattern_attribute = model_pattern.get_layer('batch_normalization_73')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_73')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_74')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_74')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_74')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_75')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_75')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_75')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_76')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_76')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_76')(pattern_attribute)

pattern_attribute_branch = model_pattern.get_layer('conv2d_71')(merge_pattern_seven)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_71')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_71')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('conv2d_72')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_72')(pattern_attribute_branch)
pattern_attribute_branch = model_pattern.get_layer('activation_72')(pattern_attribute_branch)

pattern_attribute_branch_two = model_pattern.get_layer('max_pooling2d_4')(merge_pattern_seven)

x = [pattern_attribute,pattern_attribute_branch,pattern_attribute_branch_two]

merge_pattern_eight = model_pattern.get_layer('mixed8')(x)

############################################################ mixed 9 #######################################################

pattern_attribute = model_pattern.get_layer('conv2d_81')(merge_pattern_eight)
pattern_attribute = model_pattern.get_layer('batch_normalization_81')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_81')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_82')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_82')(pattern_attribute)

pattern_attribute_branch_activation = model_pattern.get_layer('activation_82')(pattern_attribute) # connect to conv_2d_84

pattern_attribute_first_activation = model_pattern.get_layer('conv2d_83')(pattern_attribute_branch_activation)
pattern_attribute_first_activation = model_pattern.get_layer('batch_normalization_83')(pattern_attribute_first_activation)
pattern_attribute_first_activation = model_pattern.get_layer('activation_83')(pattern_attribute_first_activation)

pattern_attribute_activation = model_pattern.get_layer('conv2d_84')(pattern_attribute_branch_activation)
pattern_attribute_activation = model_pattern.get_layer('batch_normalization_84')(pattern_attribute_activation)
pattern_attribute_activation = model_pattern.get_layer('activation_84')(pattern_attribute_activation)

x = [pattern_attribute_first_activation,pattern_attribute_activation]

merge_pattern_one = model_pattern.get_layer('concatenate_1')(x)

pattern_attribute_branch = model_pattern.get_layer('conv2d_78')(merge_pattern_eight)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_78')(pattern_attribute_branch)
pattern_attribute_branch_one = model_pattern.get_layer('activation_78')(pattern_attribute_branch)

pattern_attribute_branch_sub_branch1 = model_pattern.get_layer('conv2d_79')(pattern_attribute_branch_one)
pattern_attribute_branch_sub_branch1 = model_pattern.get_layer('batch_normalization_79')(pattern_attribute_branch_sub_branch1)
pattern_attribute_branch_sub_branch1 = model_pattern.get_layer('activation_79')(pattern_attribute_branch_sub_branch1)

pattern_attribute_branch_sub_branch2 = model_pattern.get_layer('conv2d_80')(pattern_attribute_branch_one)
pattern_attribute_branch_sub_branch2 = model_pattern.get_layer('batch_normalization_80')(pattern_attribute_branch_sub_branch2)
pattern_attribute_branch_sub_branch2 = model_pattern.get_layer('activation_80')(pattern_attribute_branch_sub_branch2)

x = [pattern_attribute_branch_sub_branch1,pattern_attribute_branch_sub_branch2]

merge_pattern_two = model_pattern.get_layer('mixed9_0')(x)

pattern_attribute_branch_two = model_pattern.get_layer('average_pooling2d_8')(merge_pattern_eight)
pattern_attribute_branch_two = model_pattern.get_layer('conv2d_85')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('batch_normalization_85')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('activation_85')(pattern_attribute_branch_two)

pattern_attribute_branch_three = model_pattern.get_layer('conv2d_77')(merge_pattern_eight)
pattern_attribute_branch_three = model_pattern.get_layer('batch_normalization_77')(pattern_attribute_branch_three)
pattern_attribute_branch_three = model_pattern.get_layer('activation_77')(pattern_attribute_branch_three)

x = [merge_pattern_one,merge_pattern_two,pattern_attribute_branch_two,pattern_attribute_branch_three]

merge_pattern_nine = model_pattern.get_layer('mixed9')(x)

############################################################ Merge 10 ##################################################

pattern_attribute = model_pattern.get_layer('conv2d_90')(merge_pattern_nine)
pattern_attribute = model_pattern.get_layer('batch_normalization_90')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_90')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_91')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_91')(pattern_attribute)

pattern_attribute_branch_activation = model_pattern.get_layer('activation_91')(pattern_attribute) # connect to conv_2d_84

pattern_attribute = model_pattern.get_layer('conv2d_92')(pattern_attribute_branch_activation)
pattern_attribute = model_pattern.get_layer('batch_normalization_92')(pattern_attribute_branch_activation)
pattern_attribute = model_pattern.get_layer('activation_92')(pattern_attribute_branch_activation)

pattern_attribute_activation = model_pattern.get_layer('conv2d_93')(pattern_attribute_branch_activation)
pattern_attribute_activation = model_pattern.get_layer('batch_normalization_93')(pattern_attribute_activation)
pattern_attribute_activation = model_pattern.get_layer('activation_93')(pattern_attribute_activation)

x = [pattern_attribute_branch_activation,pattern_attribute_activation]

merge_pattern_one = model_pattern.get_layer('concatenate_2')(x)

pattern_attribute_branch = model_pattern.get_layer('conv2d_87')(merge_pattern_nine)
pattern_attribute_branch = model_pattern.get_layer('batch_normalization_87')(pattern_attribute_branch)
pattern_attribute_branch_one = model_pattern.get_layer('activation_87')(pattern_attribute_branch)

pattern_attribute_branch_sub_branch1 = model_pattern.get_layer('conv2d_88')(pattern_attribute_branch_one)
pattern_attribute_branch_sub_branch1 = model_pattern.get_layer('batch_normalization_88')(pattern_attribute_branch_sub_branch1)
pattern_attribute_branch_sub_branch1 = model_pattern.get_layer('activation_88')(pattern_attribute_branch_sub_branch1)

pattern_attribute_branch_sub_branch2 = model_pattern.get_layer('conv2d_89')(pattern_attribute_branch_one)
pattern_attribute_branch_sub_branch2 = model_pattern.get_layer('batch_normalization_89')(pattern_attribute_branch_sub_branch2)
pattern_attribute_branch_sub_branch2 = model_pattern.get_layer('activation_89')(pattern_attribute_branch_sub_branch2)

x = [pattern_attribute_branch_sub_branch1,pattern_attribute_branch_sub_branch2]

merge_pattern_two = model_pattern.get_layer('mixed9_1')(x)

pattern_attribute_branch_two = model_pattern.get_layer('average_pooling2d_9')(merge_pattern_nine)
pattern_attribute_branch_two = model_pattern.get_layer('conv2d_94')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('batch_normalization_94')(pattern_attribute_branch_two)
pattern_attribute_branch_two = model_pattern.get_layer('activation_94')(pattern_attribute_branch_two)

pattern_attribute_branch_three = model_pattern.get_layer('conv2d_86')(merge_pattern_nine)
pattern_attribute_branch_three = model_pattern.get_layer('batch_normalization_86')(pattern_attribute_branch_three)
pattern_attribute_branch_three = model_pattern.get_layer('activation_86')(pattern_attribute_branch_three)

x = [merge_pattern_one,merge_pattern_two,pattern_attribute_branch_two,pattern_attribute_branch_three]

merge_pattern_ten = model_pattern.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

pattern_attribute = model_pattern.get_layer('global_average_pooling2d_1')(merge_pattern_ten)
pattern_attribute = model_pattern.get_layer('dropout_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('attribute_pattern')(pattern_attribute)
predictions_pattern = model_pattern.get_layer('predictions_pattern')(pattern_attribute)



##########################################################################################################################
################################################## Model for color #####################################################
##########################################################################################################################

print("Preparing Color Model...")

color_attribute = model_pattern.get_layer('mixed5').output
color_attribute = model_color.get_layer('conv2d_55_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_55_color')(color_attribute)
color_attribute = model_color.get_layer('activation_55_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_56_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_56_color')(color_attribute)
color_attribute = model_color.get_layer('activation_56_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_57_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_57_color')(color_attribute)
color_attribute = model_color.get_layer('activation_57_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_58_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_58_color')(color_attribute)
color_attribute = model_color.get_layer('activation_58_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_59_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_59_color')(color_attribute)
color_attribute = model_color.get_layer('activation_59_color')(color_attribute)


color_attribute_branch = model_pattern.get_layer('mixed5').output
color_attribute_branch = model_color.get_layer('conv2d_52_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_52_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_52_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_53_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_53_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_53_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_54_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_54_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_54_color')(color_attribute_branch)


color_attribute_branch_two = model_pattern.get_layer('mixed5').output
color_attribute_branch_two = model_color.get_layer('average_pooling2d_6_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('conv2d_60_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_60_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_60_color')(color_attribute_branch_two)

color_attribute_branch_three = model_pattern.get_layer('mixed5').output
color_attribute_branch_three = model_color.get_layer('conv2d_51_color')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('batch_normalization_51_color')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_51_color')(color_attribute_branch_three)

# Merge layers
x = [color_attribute,color_attribute_branch,color_attribute_branch_two,color_attribute_branch_three]

merge_color_six = model_color.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


color_attribute = model_color.get_layer('conv2d_65')(merge_color_six)
color_attribute = model_color.get_layer('batch_normalization_65_color')(color_attribute)
color_attribute = model_color.get_layer('activation_65_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_66_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_66_color')(color_attribute)
color_attribute = model_color.get_layer('activation_66_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_67_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_67_color')(color_attribute)
color_attribute = model_color.get_layer('activation_67_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_68_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_68_color')(color_attribute)
color_attribute = model_color.get_layer('activation_68_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_69_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_69_color')(color_attribute)
color_attribute = model_color.get_layer('activation_69_color')(color_attribute)


color_attribute_branch = model_color.get_layer('conv2d_62_color')(merge_color_six)
color_attribute_branch = model_color.get_layer('batch_normalization_62_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_62_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_63_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_63_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_63_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_64_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_64_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_64_color')(color_attribute_branch)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_7_color')(merge_color_six)
color_attribute_branch_two = model_color.get_layer('conv2d_70_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_70_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_70_color')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_61_color')(merge_color_six)
color_attribute_branch_three = model_color.get_layer('batch_normalization_61_color')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_61_color')(color_attribute_branch_three)

# Merge layers
x = [color_attribute,color_attribute_branch,color_attribute_branch_two,color_attribute_branch_three]

merge_color_seven = model_color.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

color_attribute = model_color.get_layer('conv2d_73_color')(merge_color_seven)
color_attribute = model_color.get_layer('batch_normalization_73_color')(color_attribute)
color_attribute = model_color.get_layer('activation_73_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_74_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_74_color')(color_attribute)
color_attribute = model_color.get_layer('activation_74_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_75_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_75_color')(color_attribute)
color_attribute = model_color.get_layer('activation_75_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_76_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_76_color')(color_attribute)
color_attribute = model_color.get_layer('activation_76_color')(color_attribute)

color_attribute_branch = model_color.get_layer('conv2d_71_color')(merge_color_seven)
color_attribute_branch = model_color.get_layer('batch_normalization_71_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_71_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_72_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_72_color')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_72_color')(color_attribute_branch)

color_attribute_branch_two = model_color.get_layer('max_pooling2d_4_color')(merge_color_seven)

x = [color_attribute,color_attribute_branch,color_attribute_branch_two]

merge_color_eight = model_color.get_layer('mixed8_color')(x)

############################################################ mixed 9 #######################################################

color_attribute = model_color.get_layer('conv2d_81_color')(merge_color_eight)
color_attribute = model_color.get_layer('batch_normalization_81_color')(color_attribute)
color_attribute = model_color.get_layer('activation_81_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_82_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_82_color')(color_attribute)

color_attribute_branch_activation = model_color.get_layer('activation_82_color')(color_attribute) # connect to conv_2d_84

color_attribute_first_activation = model_color.get_layer('conv2d_83_color')(color_attribute_branch_activation)
color_attribute_first_activation = model_color.get_layer('batch_normalization_83_color')(color_attribute_first_activation)
color_attribute_first_activation = model_color.get_layer('activation_83_color')(color_attribute_first_activation)
color_attribute_activation = model_color.get_layer('conv2d_84_color')(color_attribute_branch_activation)
color_attribute_activation = model_color.get_layer('batch_normalization_84_color')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('activation_84_color')(color_attribute_activation)

x = [color_attribute_first_activation,color_attribute_activation]

merge_color_one = model_color.get_layer('concatenate_1_color')(x)

color_attribute_branch = model_color.get_layer('conv2d_78_color')(merge_color_eight)
color_attribute_branch = model_color.get_layer('batch_normalization_78_color')(color_attribute_branch)
color_attribute_branch_one = model_color.get_layer('activation_78_color')(color_attribute_branch)

color_attribute_branch_sub_branch1 = model_color.get_layer('conv2d_79_color')(color_attribute_branch_one)
color_attribute_branch_sub_branch1 = model_color.get_layer('batch_normalization_79_color')(color_attribute_branch_sub_branch1)
color_attribute_branch_sub_branch1 = model_color.get_layer('activation_79_color')(color_attribute_branch_sub_branch1)

color_attribute_branch_sub_branch2 = model_color.get_layer('conv2d_80_color')(color_attribute_branch_one)
color_attribute_branch_sub_branch2 = model_color.get_layer('batch_normalization_80_color')(color_attribute_branch_sub_branch2)
color_attribute_branch_sub_branch2 = model_color.get_layer('activation_80_color')(color_attribute_branch_sub_branch2)

x = [color_attribute_branch_sub_branch1,color_attribute_branch_sub_branch2]

merge_color_two = model_color.get_layer('mixed9_0_color')(x)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_8_color')(merge_color_eight)
color_attribute_branch_two = model_color.get_layer('conv2d_85_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_85_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_85_color')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_77_color')(merge_color_eight)
color_attribute_branch_three = model_color.get_layer('batch_normalization_77_color')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_77_color')(color_attribute_branch_three)

x = [merge_color_one,merge_color_two,color_attribute_branch_two,color_attribute_branch_three]

merge_color_nine = model_color.get_layer('mixed9_color')(x)

############################################################ Merge 10 ##################################################

color_attribute = model_color.get_layer('conv2d_90_color')(merge_color_nine)
color_attribute = model_color.get_layer('batch_normalization_90_color')(color_attribute)
color_attribute = model_color.get_layer('activation_90_color')(color_attribute)
color_attribute = model_color.get_layer('conv2d_91_color')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_91_color')(color_attribute)

color_attribute_branch_activation = model_color.get_layer('activation_91_color')(color_attribute) # connect to conv_2d_84

color_attribute = model_color.get_layer('conv2d_92_color')(color_attribute_branch_activation)
color_attribute = model_color.get_layer('batch_normalization_92_color')(color_attribute_branch_activation)
color_attribute = model_color.get_layer('activation_92_color')(color_attribute_branch_activation)

color_attribute_activation = model_color.get_layer('conv2d_93_color')(color_attribute_branch_activation)
color_attribute_activation = model_color.get_layer('batch_normalization_93_color')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('activation_93_color')(color_attribute_activation)

x = [color_attribute_branch_activation,color_attribute_activation]

merge_color_one = model_color.get_layer('concatenate_2_color')(x)

color_attribute_branch = model_color.get_layer('conv2d_87_color')(merge_color_nine)
color_attribute_branch = model_color.get_layer('batch_normalization_87_color')(color_attribute_branch)
color_attribute_branch_one = model_color.get_layer('activation_87_color')(color_attribute_branch)

color_attribute_branch_sub_branch1 = model_color.get_layer('conv2d_88_color')(color_attribute_branch_one)
color_attribute_branch_sub_branch1 = model_color.get_layer('batch_normalization_88_color')(color_attribute_branch_sub_branch1)
color_attribute_branch_sub_branch1 = model_color.get_layer('activation_88_color')(color_attribute_branch_sub_branch1)

color_attribute_branch_sub_branch2 = model_color.get_layer('conv2d_89_color')(color_attribute_branch_one)
color_attribute_branch_sub_branch2 = model_color.get_layer('batch_normalization_89_color')(color_attribute_branch_sub_branch2)
color_attribute_branch_sub_branch2 = model_color.get_layer('activation_89_color')(color_attribute_branch_sub_branch2)

x = [color_attribute_branch_sub_branch1,color_attribute_branch_sub_branch2]

merge_color_two = model_color.get_layer('mixed9_1_color')(x)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_9_color')(merge_color_nine)
color_attribute_branch_two = model_color.get_layer('conv2d_94_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_94_color')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_94_color')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_86_color')(merge_color_nine)
color_attribute_branch_three = model_color.get_layer('batch_normalization_86_color')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_86_color')(color_attribute_branch_three)

x = [merge_color_one,merge_color_two,color_attribute_branch_two,color_attribute_branch_three]

merge_color_ten = model_color.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

color_attribute = model_color.get_layer('global_average_pooling2d_1_color')(merge_color_ten)
color_attribute = model_color.get_layer('dropout_1_color')(color_attribute)
color_attribute = model_color.get_layer('attribute_color_color')(color_attribute)
predictions_color = model_color.get_layer('predictions_color_color')(color_attribute)



##########################################################################################################################
################################################## Model for gender #####################################################
##########################################################################################################################

##########################################################################################################################
################################################## Model for gender #####################################################
##########################################################################################################################

print("Preparing gender Model...")

gender_attribute = model_pattern.get_layer('mixed5').output
gender_attribute = model_gender.get_layer('conv2d_55_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_55_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_55_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_56_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_56_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_56_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_57_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_57_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_57_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_58_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_58_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_58_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_59_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_59_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_59_gender')(gender_attribute)


gender_attribute_branch = model_pattern.get_layer('mixed5').output
gender_attribute_branch = model_gender.get_layer('conv2d_52_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_52_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_52_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_53_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_53_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_53_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_54_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_54_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_54_gender')(gender_attribute_branch)


gender_attribute_branch_two = model_pattern.get_layer('mixed5').output
gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_6_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('conv2d_60_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_60_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_60_gender')(gender_attribute_branch_two)

gender_attribute_branch_three = model_pattern.get_layer('mixed5').output
gender_attribute_branch_three = model_gender.get_layer('conv2d_51_gender')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_51_gender')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_51_gender')(gender_attribute_branch_three)

# Merge layers
x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_six = model_gender.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


gender_attribute = model_gender.get_layer('conv2d_65')(merge_gender_six)
gender_attribute = model_gender.get_layer('batch_normalization_65_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_65_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_66_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_66_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_66_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_67_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_67_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_67_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_68_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_68_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_68_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_69_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_69_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_69_gender')(gender_attribute)


gender_attribute_branch = model_gender.get_layer('conv2d_62_gender')(merge_gender_six)
gender_attribute_branch = model_gender.get_layer('batch_normalization_62_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_62_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_63_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_63_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_63_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_64_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_64_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_64_gender')(gender_attribute_branch)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_7_gender')(merge_gender_six)
gender_attribute_branch_two = model_gender.get_layer('conv2d_70_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_70_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_70_gender')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_61_gender')(merge_gender_six)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_61_gender')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_61_gender')(gender_attribute_branch_three)

# Merge layers
x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_seven = model_gender.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

gender_attribute = model_gender.get_layer('conv2d_73_gender')(merge_gender_seven)
gender_attribute = model_gender.get_layer('batch_normalization_73_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_73_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_74_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_74_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_74_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_75_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_75_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_75_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_76_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_76_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_76_gender')(gender_attribute)

gender_attribute_branch = model_gender.get_layer('conv2d_71_gender')(merge_gender_seven)
gender_attribute_branch = model_gender.get_layer('batch_normalization_71_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_71_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_72_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_72_gender')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_72_gender')(gender_attribute_branch)

gender_attribute_branch_two = model_gender.get_layer('max_pooling2d_4_gender')(merge_gender_seven)

x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two]

merge_gender_eight = model_gender.get_layer('mixed8_gender')(x)

############################################################ mixed 9 #######################################################

gender_attribute = model_gender.get_layer('conv2d_81_gender')(merge_gender_eight)
gender_attribute = model_gender.get_layer('batch_normalization_81_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_81_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_82_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_82_gender')(gender_attribute)

gender_attribute_branch_activation = model_gender.get_layer('activation_82_gender')(gender_attribute) # connect to conv_2d_84

gender_attribute_first_activation = model_gender.get_layer('conv2d_83_gender')(gender_attribute_branch_activation)
gender_attribute_first_activation = model_gender.get_layer('batch_normalization_83_gender')(gender_attribute_first_activation)
gender_attribute_first_activation = model_gender.get_layer('activation_83_gender')(gender_attribute_first_activation)
gender_attribute_activation = model_gender.get_layer('conv2d_84_gender')(gender_attribute_branch_activation)
gender_attribute_activation = model_gender.get_layer('batch_normalization_84_gender')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('activation_84_gender')(gender_attribute_activation)

x = [gender_attribute_first_activation,gender_attribute_activation]

merge_gender_one = model_gender.get_layer('concatenate_1_gender')(x)

gender_attribute_branch = model_gender.get_layer('conv2d_78_gender')(merge_gender_eight)
gender_attribute_branch = model_gender.get_layer('batch_normalization_78_gender')(gender_attribute_branch)
gender_attribute_branch_one = model_gender.get_layer('activation_78_gender')(gender_attribute_branch)

gender_attribute_branch_sub_branch1 = model_gender.get_layer('conv2d_79_gender')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('batch_normalization_79_gender')(gender_attribute_branch_sub_branch1)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('activation_79_gender')(gender_attribute_branch_sub_branch1)

gender_attribute_branch_sub_branch2 = model_gender.get_layer('conv2d_80_gender')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('batch_normalization_80_gender')(gender_attribute_branch_sub_branch2)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('activation_80_gender')(gender_attribute_branch_sub_branch2)

x = [gender_attribute_branch_sub_branch1,gender_attribute_branch_sub_branch2]

merge_gender_two = model_gender.get_layer('mixed9_0_gender')(x)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_8_gender')(merge_gender_eight)
gender_attribute_branch_two = model_gender.get_layer('conv2d_85_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_85_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_85_gender')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_77_gender')(merge_gender_eight)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_77_gender')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_77_gender')(gender_attribute_branch_three)

x = [merge_gender_one,merge_gender_two,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_nine = model_gender.get_layer('mixed9_gender')(x)

############################################################ Merge 10 ##################################################

gender_attribute = model_gender.get_layer('conv2d_90_gender')(merge_gender_nine)
gender_attribute = model_gender.get_layer('batch_normalization_90_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_90_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_91_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_91_gender')(gender_attribute)

gender_attribute_branch_activation = model_gender.get_layer('activation_91_gender')(gender_attribute) # connect to conv_2d_84

gender_attribute = model_gender.get_layer('conv2d_92_gender')(gender_attribute_branch_activation)
gender_attribute = model_gender.get_layer('batch_normalization_92_gender')(gender_attribute_branch_activation)
gender_attribute = model_gender.get_layer('activation_92_gender')(gender_attribute_branch_activation)

gender_attribute_activation = model_gender.get_layer('conv2d_93_gender')(gender_attribute_branch_activation)
gender_attribute_activation = model_gender.get_layer('batch_normalization_93_gender')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('activation_93_gender')(gender_attribute_activation)

x = [gender_attribute_branch_activation,gender_attribute_activation]

merge_gender_one = model_gender.get_layer('concatenate_2_gender')(x)

gender_attribute_branch = model_gender.get_layer('conv2d_87_gender')(merge_gender_nine)
gender_attribute_branch = model_gender.get_layer('batch_normalization_87_gender')(gender_attribute_branch)
gender_attribute_branch_one = model_gender.get_layer('activation_87_gender')(gender_attribute_branch)

gender_attribute_branch_sub_branch1 = model_gender.get_layer('conv2d_88_gender')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('batch_normalization_88_gender')(gender_attribute_branch_sub_branch1)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('activation_88_gender')(gender_attribute_branch_sub_branch1)

gender_attribute_branch_sub_branch2 = model_gender.get_layer('conv2d_89_gender')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('batch_normalization_89_gender')(gender_attribute_branch_sub_branch2)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('activation_89_gender')(gender_attribute_branch_sub_branch2)

x = [gender_attribute_branch_sub_branch1,gender_attribute_branch_sub_branch2]

merge_gender_two = model_gender.get_layer('mixed9_1_gender')(x)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_9_gender')(merge_gender_nine)
gender_attribute_branch_two = model_gender.get_layer('conv2d_94_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_94_gender')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_94_gender')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_86_gender')(merge_gender_nine)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_86_gender')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_86_gender')(gender_attribute_branch_three)

x = [merge_gender_one,merge_gender_two,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_ten = model_gender.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

gender_attribute = model_gender.get_layer('global_average_pooling2d_1_gender')(merge_gender_ten)
gender_attribute = model_gender.get_layer('dropout_1_gender')(gender_attribute)
gender_attribute = model_gender.get_layer('attribute_gender_gender')(gender_attribute)
predictions_gender = model_gender.get_layer('predictions_gender_gender')(gender_attribute)

final_model = Model(inputs = model_pattern.input, outputs= [predictions_pattern,predictions_color,predictions_gender])
final_model.save("../models/final_model.h5")
print("Model Created.")