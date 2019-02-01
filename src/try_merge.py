from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model

model_pattern = load_model("../models/finalmodel/label_pattern_inceptionv3_50_0.54.h5")
model_color = load_model("../models/finalmodel/label_color_inceptionv3_49_0.35.h5")
model_gender = load_model("../models/finalmodel/label_gender_inceptionv3_08_0.83.h5")

print("Preparing Pattern Model.")

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
pattern_attribute_activation = model_pattern.get_layer('activation_82')(pattern_attribute) # connect to conv_2d_84
pattern_attribute = model_pattern.get_layer('conv2d_83')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_83')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_83')(pattern_attribute)

pattern_attribute_activation = model_pattern.get_layer('conv2d_84')(pattern_attribute_activation)
pattern_attribute_activation = model_pattern.get_layer('batch_normalization_84')(pattern_attribute_activation)
pattern_attribute_activation = model_pattern.get_layer('activation_84')(pattern_attribute_activation)

x = [pattern_attribute,pattern_attribute_activation]

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

pattern_attribute_branch_three = model_pattern.get_layer('conv2d_85')(merge_pattern_eight)
pattern_attribute_branch_three = model_pattern.get_layer('batch_normalization_85')(pattern_attribute_branch_three)
pattern_attribute_branch_three = model_pattern.get_layer('activation_85')(pattern_attribute_branch_three)

x = [merge_pattern_one,merge_pattern_two,pattern_attribute_branch_two,pattern_attribute_branch_three]

merge_pattern_nine = model_pattern.get_layer('mixed9')(x)

############################################################ Merge 10 ##################################################

pattern_attribute = model_pattern.get_layer('conv2d_90')(merge_pattern_nine)
pattern_attribute = model_pattern.get_layer('batch_normalization_90')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_90')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('conv2d_91')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_91')(pattern_attribute)
pattern_attribute_activation = model_pattern.get_layer('activation_91')(pattern_attribute) # connect to conv_2d_84
pattern_attribute = model_pattern.get_layer('conv2d_92')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('batch_normalization_92')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('activation_92')(pattern_attribute)

pattern_attribute_activation = model_pattern.get_layer('conv2d_93')(pattern_attribute_activation)
pattern_attribute_activation = model_pattern.get_layer('batch_normalization_93')(pattern_attribute_activation)
pattern_attribute_activation = model_pattern.get_layer('activation_93')(pattern_attribute_activation)

x = [pattern_attribute,pattern_attribute_activation]

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

pattern_attribute = model_pattern.get_layer('mixed10')(merge_pattern_ten)
pattern_attribute = model_pattern.get_layer('global_average_pooling2d_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('dropout_1')(pattern_attribute)
pattern_attribute = model_pattern.get_layer('attribute_pattern')(pattern_attribute)
predictions_pattern = model_pattern.get_layer('predictions_pattern')(pattern_attribute)



##########################################################################################################################
################################################## Model for color #####################################################
##########################################################################################################################

print("Preparing Color Model...")

color_attribute = model_pattern.get_layer('mixed10').output
color_attribute = model_color.get_layer('conv2d_55')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_55')(color_attribute)
color_attribute = model_color.get_layer('activation_55')(color_attribute)
color_attribute = model_color.get_layer('conv2d_56')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_56')(color_attribute)
color_attribute = model_color.get_layer('activation_56')(color_attribute)
color_attribute = model_color.get_layer('conv2d_57')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_57')(color_attribute)
color_attribute = model_color.get_layer('activation_57')(color_attribute)
color_attribute = model_color.get_layer('conv2d_58')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_58')(color_attribute)
color_attribute = model_color.get_layer('activation_58')(color_attribute)
color_attribute = model_color.get_layer('conv2d_59')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_59')(color_attribute)
color_attribute = model_color.get_layer('activation_59')(color_attribute)


color_attribute_branch = model_color.get_layer('mixed5').output
color_attribute_branch = model_color.get_layer('conv2d_52')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_52')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_52')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_53')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_53')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_53')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_54')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_54')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_54')(color_attribute_branch)


color_attribute_branch_two = model_color.get_layer('mixed5').output
color_attribute_branch_two = model_color.get_layer('average_pooling2d_6')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('conv2d_60')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_60')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_60')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('mixed5').output
color_attribute_branch_three = model_color.get_layer('conv2d_51')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('batch_normalization_51')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_51')(color_attribute_branch_three)

# Merge layers
x = [color_attribute,color_attribute_branch,color_attribute_branch_two,color_attribute_branch_three]

merge_color_six = model_color.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


color_attribute = model_color.get_layer('conv2d_65')(merge_color_six)
color_attribute = model_color.get_layer('batch_normalization_65')(color_attribute)
color_attribute = model_color.get_layer('activation_65')(color_attribute)
color_attribute = model_color.get_layer('conv2d_66')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_66')(color_attribute)
color_attribute = model_color.get_layer('activation_66')(color_attribute)
color_attribute = model_color.get_layer('conv2d_67')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_67')(color_attribute)
color_attribute = model_color.get_layer('activation_67')(color_attribute)
color_attribute = model_color.get_layer('conv2d_68')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_68')(color_attribute)
color_attribute = model_color.get_layer('activation_68')(color_attribute)
color_attribute = model_color.get_layer('conv2d_69')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_69')(color_attribute)
color_attribute = model_color.get_layer('activation_69')(color_attribute)


color_attribute_branch = model_color.get_layer('conv2d_62')(merge_color_six)
color_attribute_branch = model_color.get_layer('batch_normalization_62')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_62')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_63')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_63')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_63')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_64')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_64')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_64')(color_attribute_branch)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_7')(merge_color_six)
color_attribute_branch_two = model_color.get_layer('conv2d_70')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_70')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_70')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_61')(merge_color_six)
color_attribute_branch_three = model_color.get_layer('batch_normalization_61')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_61')(color_attribute_branch_three)

# Merge layers
x = [color_attribute,color_attribute_branch,color_attribute_branch_two,color_attribute_branch_three]

merge_color_seven = model_color.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

color_attribute = model_color.get_layer('conv2d_73')(merge_color_seven)
color_attribute = model_color.get_layer('batch_normalization_73')(color_attribute)
color_attribute = model_color.get_layer('activation_73')(color_attribute)
color_attribute = model_color.get_layer('conv2d_74')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_74')(color_attribute)
color_attribute = model_color.get_layer('activation_74')(color_attribute)
color_attribute = model_color.get_layer('conv2d_75')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_75')(color_attribute)
color_attribute = model_color.get_layer('activation_75')(color_attribute)
color_attribute = model_color.get_layer('conv2d_76')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_76')(color_attribute)
color_attribute = model_color.get_layer('activation_76')(color_attribute)

color_attribute_branch = model_color.get_layer('conv2d_71')(merge_color_seven)
color_attribute_branch = model_color.get_layer('batch_normalization_71')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_71')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_72')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_72')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_72')(color_attribute_branch)

color_attribute_branch_two = model_color.get_layer('max_pooling2d_4')(merge_color_seven)

x = [color_attribute,color_attribute_branch,color_attribute_branch_two]

merge_color_eight = model_color.get_layer('mixed8')(x)

############################################################ mixed 9 #######################################################


color_attribute = model_color.get_layer('conv2d_81')(merge_color_eight)
color_attribute = model_color.get_layer('batch_normalization_81')(color_attribute)
color_attribute = model_color.get_layer('activation_81')(color_attribute)
color_attribute = model_color.get_layer('conv2d_82')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_82')(color_attribute)
color_attribute_activation = model_color.get_layer('activation_82')(color_attribute) # connect to conv_2d_84
color_attribute = model_color.get_layer('conv2d_83')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_83')(color_attribute)
color_attribute = model_color.get_layer('activation_83')(color_attribute)

color_attribute_activation = model_color.get_layer('conv2d_84')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('batch_normalization_84')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('activation_84')(color_attribute_activation)

x = [color_attribute,color_attribute_activation]

merge_color_one = model_color.get_layer('concatenate_1')(x)

color_attribute_branch = model_color.get_layer('conv2d_78')(merge_color_eight)
color_attribute_branch = model_color.get_layer('batch_normalization_78')(color_attribute_branch)
color_attribute_branch_one = model_color.get_layer('activation_78')(color_attribute_branch)

color_attribute_branch_sub_branch1 = model_color.get_layer('conv2d_79')(color_attribute_branch_one)
color_attribute_branch_sub_branch1 = model_color.get_layer('batch_normalization_79')(color_attribute_branch_sub_branch1)
color_attribute_branch_sub_branch1 = model_color.get_layer('activation_79')(color_attribute_branch_sub_branch1)

color_attribute_branch_sub_branch2 = model_color.get_layer('conv2d_80')(color_attribute_branch_one)
color_attribute_branch_sub_branch2 = model_color.get_layer('batch_normalization_80')(color_attribute_branch_sub_branch2)
color_attribute_branch_sub_branch2 = model_color.get_layer('activation_80')(color_attribute_branch_sub_branch2)

x = [color_attribute_branch_sub_branch1,color_attribute_branch_sub_branch2]

merge_color_two = model_color.get_layer('mixed9_0')(x)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_8')(merge_color_eight)
color_attribute_branch_two = model_color.get_layer('conv2d_85')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_85')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_85')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_85')(merge_color_eight)
color_attribute_branch_three = model_color.get_layer('batch_normalization_85')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_85')(color_attribute_branch_three)

x = [merge_color_one,merge_color_two,color_attribute_branch_two,color_attribute_branch_three]

merge_color_nine = model_color.get_layer('mixed9')(x)

############################################################ Merge 10 ##################################################

color_attribute = model_color.get_layer('conv2d_90')(merge_color_nine)
color_attribute = model_color.get_layer('batch_normalization_90')(color_attribute)
color_attribute = model_color.get_layer('activation_90')(color_attribute)
color_attribute = model_color.get_layer('conv2d_91')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_91')(color_attribute)
color_attribute_activation = model_color.get_layer('activation_91')(color_attribute) # connect to conv_2d_84
color_attribute = model_color.get_layer('conv2d_92')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_92')(color_attribute)
color_attribute = model_color.get_layer('activation_92')(color_attribute)

color_attribute_activation = model_color.get_layer('conv2d_93')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('batch_normalization_93')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('activation_93')(color_attribute_activation)

x = [color_attribute,color_attribute_activation]

merge_color_one = model_color.get_layer('concatenate_2')(x)

color_attribute_branch = model_color.get_layer('conv2d_87')(merge_color_nine)
color_attribute_branch = model_color.get_layer('batch_normalization_87')(color_attribute_branch)
color_attribute_branch_one = model_color.get_layer('activation_87')(color_attribute_branch)

color_attribute_branch_sub_branch1 = model_color.get_layer('conv2d_88')(color_attribute_branch_one)
color_attribute_branch_sub_branch1 = model_color.get_layer('batch_normalization_88')(color_attribute_branch_sub_branch1)
color_attribute_branch_sub_branch1 = model_color.get_layer('activation_88')(color_attribute_branch_sub_branch1)

color_attribute_branch_sub_branch2 = model_color.get_layer('conv2d_89')(color_attribute_branch_one)
color_attribute_branch_sub_branch2 = model_color.get_layer('batch_normalization_89')(color_attribute_branch_sub_branch2)
color_attribute_branch_sub_branch2 = model_color.get_layer('activation_89')(color_attribute_branch_sub_branch2)

x = [color_attribute_branch_sub_branch1,color_attribute_branch_sub_branch2]

merge_color_two = model_color.get_layer('mixed9_1')(x)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_9')(merge_color_nine)
color_attribute_branch_two = model_color.get_layer('conv2d_94')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_94')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_94')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_86')(merge_color_nine)
color_attribute_branch_three = model_color.get_layer('batch_normalization_86')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_86')(color_attribute_branch_three)

x = [merge_color_one,merge_color_two,color_attribute_branch_two,color_attribute_branch_three]

merge_color_ten = model_color.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

color_attribute = model_color.get_layer('mixed10')(merge_color_ten)
color_attribute = model_color.get_layer('global_average_pooling2d_1')(color_attribute)
color_attribute = model_color.get_layer('dropout_1')(color_attribute)
color_attribute = model_color.get_layer('attribute_color')(color_attribute)
predictions_color = model_color.get_layer('predictions_color')(color_attribute)



##########################################################################################################################
################################################## Model for gender #####################################################
##########################################################################################################################

print("Preparing Gender Model.")

gender_attribute = model_pattern.get_layer('mixed10').output
gender_attribute = model_gender.get_layer('conv2d_55')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_55')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_55')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_56')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_56')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_56')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_57')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_57')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_57')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_58')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_58')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_58')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_59')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_59')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_59')(gender_attribute)


gender_attribute_branch = model_gender.get_layer('mixed5').output
gender_attribute_branch = model_gender.get_layer('conv2d_52')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_52')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_52')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_53')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_53')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_53')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_54')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_54')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_54')(gender_attribute_branch)


gender_attribute_branch_two = model_gender.get_layer('mixed5').output
gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_6')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('conv2d_60')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_60')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_60')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('mixed5').output
gender_attribute_branch_three = model_gender.get_layer('conv2d_51')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_51')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_51')(gender_attribute_branch_three)

# Merge layers
x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_six = model_gender.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


gender_attribute = model_gender.get_layer('conv2d_65')(merge_gender_six)
gender_attribute = model_gender.get_layer('batch_normalization_65')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_65')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_66')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_66')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_66')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_67')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_67')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_67')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_68')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_68')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_68')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_69')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_69')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_69')(gender_attribute)


gender_attribute_branch = model_gender.get_layer('conv2d_62')(merge_gender_six)
gender_attribute_branch = model_gender.get_layer('batch_normalization_62')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_62')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_63')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_63')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_63')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_64')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_64')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_64')(gender_attribute_branch)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_7')(merge_gender_six)
gender_attribute_branch_two = model_gender.get_layer('conv2d_70')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_70')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_70')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_61')(merge_gender_six)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_61')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_61')(gender_attribute_branch_three)

# Merge layers
x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_seven = model_gender.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

gender_attribute = model_gender.get_layer('conv2d_73')(merge_gender_seven)
gender_attribute = model_gender.get_layer('batch_normalization_73')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_73')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_74')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_74')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_74')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_75')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_75')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_75')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_76')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_76')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_76')(gender_attribute)

gender_attribute_branch = model_gender.get_layer('conv2d_71')(merge_gender_seven)
gender_attribute_branch = model_gender.get_layer('batch_normalization_71')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_71')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_72')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_72')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_72')(gender_attribute_branch)

gender_attribute_branch_two = model_gender.get_layer('max_pooling2d_4')(merge_gender_seven)

x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two]

merge_gender_eight = model_gender.get_layer('mixed8')(x)

############################################################ mixed 9 #######################################################


gender_attribute = model_gender.get_layer('conv2d_81')(merge_gender_eight)
gender_attribute = model_gender.get_layer('batch_normalization_81')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_81')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_82')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_82')(gender_attribute)
gender_attribute_activation = model_gender.get_layer('activation_82')(gender_attribute) # connect to conv_2d_84
gender_attribute = model_gender.get_layer('conv2d_83')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_83')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_83')(gender_attribute)

gender_attribute_activation = model_gender.get_layer('conv2d_84')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('batch_normalization_84')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('activation_84')(gender_attribute_activation)

x = [gender_attribute,gender_attribute_activation]

merge_gender_one = model_gender.get_layer('concatenate_1')(x)

gender_attribute_branch = model_gender.get_layer('conv2d_78')(merge_gender_eight)
gender_attribute_branch = model_gender.get_layer('batch_normalization_78')(gender_attribute_branch)
gender_attribute_branch_one = model_gender.get_layer('activation_78')(gender_attribute_branch)

gender_attribute_branch_sub_branch1 = model_gender.get_layer('conv2d_79')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('batch_normalization_79')(gender_attribute_branch_sub_branch1)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('activation_79')(gender_attribute_branch_sub_branch1)

gender_attribute_branch_sub_branch2 = model_gender.get_layer('conv2d_80')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('batch_normalization_80')(gender_attribute_branch_sub_branch2)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('activation_80')(gender_attribute_branch_sub_branch2)

x = [gender_attribute_branch_sub_branch1,gender_attribute_branch_sub_branch2]

merge_gender_two = model_gender.get_layer('mixed9_0')(x)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_8')(merge_gender_eight)
gender_attribute_branch_two = model_gender.get_layer('conv2d_85')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_85')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_85')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_85')(merge_gender_eight)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_85')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_85')(gender_attribute_branch_three)

x = [merge_gender_one,merge_gender_two,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_nine = model_gender.get_layer('mixed9')(x)

############################################################ Merge 10 ##################################################

gender_attribute = model_gender.get_layer('conv2d_90')(merge_gender_nine)
gender_attribute = model_gender.get_layer('batch_normalization_90')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_90')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_91')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_91')(gender_attribute)
gender_attribute_activation = model_gender.get_layer('activation_91')(gender_attribute) # connect to conv_2d_84
gender_attribute = model_gender.get_layer('conv2d_92')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_92')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_92')(gender_attribute)

gender_attribute_activation = model_gender.get_layer('conv2d_93')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('batch_normalization_93')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('activation_93')(gender_attribute_activation)

x = [gender_attribute,gender_attribute_activation]

merge_gender_one = model_gender.get_layer('concatenate_2')(x)

gender_attribute_branch = model_gender.get_layer('conv2d_87')(merge_gender_nine)
gender_attribute_branch = model_gender.get_layer('batch_normalization_87')(gender_attribute_branch)
gender_attribute_branch_one = model_gender.get_layer('activation_87')(gender_attribute_branch)

gender_attribute_branch_sub_branch1 = model_gender.get_layer('conv2d_88')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('batch_normalization_88')(gender_attribute_branch_sub_branch1)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('activation_88')(gender_attribute_branch_sub_branch1)

gender_attribute_branch_sub_branch2 = model_gender.get_layer('conv2d_89')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('batch_normalization_89')(gender_attribute_branch_sub_branch2)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('activation_89')(gender_attribute_branch_sub_branch2)

x = [gender_attribute_branch_sub_branch1,gender_attribute_branch_sub_branch2]

merge_gender_two = model_gender.get_layer('mixed9_1')(x)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_9')(merge_gender_nine)
gender_attribute_branch_two = model_gender.get_layer('conv2d_94')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_94')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_94')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_86')(merge_gender_nine)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_86')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_86')(gender_attribute_branch_three)

x = [merge_gender_one,merge_gender_two,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_ten = model_gender.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

gender_attribute = model_gender.get_layer('mixed10')(merge_gender_ten)
gender_attribute = model_gender.get_layer('global_average_pooling2d_1')(gender_attribute)
gender_attribute = model_gender.get_layer('dropout_1')(gender_attribute)
gender_attribute = model_gender.get_layer('attribute_gender')(gender_attribute)
predictions_gender = model_gender.get_layer('predictions_gender')(gender_attribute)

final_model = Model(inputs= model_pattern.input, outputs= [predictions_pattern,predictions_color])
final_model.save("../models/final_model.h5")
print("Model Created.")
