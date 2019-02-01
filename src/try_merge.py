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
color_attribute = model_color.get_layer('conv2d_55').name('conv2d_55_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_55').name('batch_normalization_55_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_55').name('activation_55_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_56').name('conv2d_56_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_56').name('batch_normalization_56_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_56').name('activation_56_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_57').name('conv2d_57_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_57').name('batch_normalization_57_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_57').name('activation_57_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_58').name('conv2d_58_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_58').name('batch_normalization_58_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_58').name('activation_58_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_59').name('conv2d_59_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_59').name('batch_normalization_59_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_59').name('activation_59_xyz')(color_attribute)


color_attribute_branch = model_pattern.get_layer('mixed5').output
color_attribute_branch = model_color.get_layer('conv2d_52').name('conv2d_52_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_52').name('batch_normalization_52_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_52').name('activation_52_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_53').name('conv2d_53_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_53').name('batch_normalization_53_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_53').name('activation_53_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_54').name('conv2d_54_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_54').name('batch_normalization_54_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_54').name('activation_54_xyz')(color_attribute_branch)


color_attribute_branch_two = model_pattern.get_layer('mixed5').output
color_attribute_branch_two = model_color.get_layer('average_pooling2d_6').name('average_pooling2d_6_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('conv2d_60').name('conv2d_60_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_60').name('batch_normalization_60_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_60').name('activation_60_xyz')(color_attribute_branch_two)

color_attribute_branch_three = model_pattern.get_layer('mixed5').output
color_attribute_branch_three = model_color.get_layer('conv2d_51').name('conv2d_51_xyz')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('batch_normalization_51').name('batch_normalization_51_xyz')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_51').name('activation_51_xyz')(color_attribute_branch_three)

# Merge layers
x = [color_attribute,color_attribute_branch,color_attribute_branch_two,color_attribute_branch_three]

merge_color_six = model_color.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


color_attribute = model_color.get_layer('conv2d_65').name('conv2d_65_xyz')(merge_color_six)
color_attribute = model_color.get_layer('batch_normalization_65').name('batch_normalization_65_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_65').name('activation_65_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_66').name('conv2d_66_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_66').name('batch_normalization_66_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_66').name('activation_66_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_67').name('conv2d_67_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_67').name('batch_normalization_67_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_67').name('activation_67_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_68').name('conv2d_68_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_68').name('batch_normalization_68_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_68').name('activation_68_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_69').name('conv2d_69_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_69').name('batch_normalization_69_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_69').name('activation_69_xyz')(color_attribute)


color_attribute_branch = model_color.get_layer('conv2d_62').name('conv2d_62_xyz')(merge_color_six)
color_attribute_branch = model_color.get_layer('batch_normalization_62').name('batch_normalization_62_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_62').name('activation_62_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_63').name('conv2d_63_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_63').name('batch_normalization_63_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_63').name('activation_63_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_64').name('conv2d_64_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_64').name('batch_normalization_64_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_64').name('activation_64_xyz')(color_attribute_branch)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_7').name('average_pooling2d_7_xyz')(merge_color_six)
color_attribute_branch_two = model_color.get_layer('conv2d_70').name('conv2d_70_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_70').name('batch_normalization_70_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_70').name('activation_70_xyz')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_61').name('conv2d_61_xyz')(merge_color_six)
color_attribute_branch_three = model_color.get_layer('batch_normalization_61').name('batch_normalization_61_xyz')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_61').name('activation_61_xyz')(color_attribute_branch_three)

# Merge layers
x = [color_attribute,color_attribute_branch,color_attribute_branch_two,color_attribute_branch_three]

merge_color_seven = model_color.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

color_attribute = model_color.get_layer('conv2d_73').name('conv2d_73_xyz')(merge_color_seven)
color_attribute = model_color.get_layer('batch_normalization_73').name('batch_normalization_73_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_73').name('activation_73_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_74').name('conv2d_74_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_74').name('batch_normalization_74_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_74').name('activation_74_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_75').name('conv2d_75_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_75').name('batch_normalization_75_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_75').name('activation_75_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_76').name('conv2d_76_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_76').name('batch_normalization_76_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_76').name('activation_76_xyz')(color_attribute)

color_attribute_branch = model_color.get_layer('conv2d_71').name('conv2d_71_xyz')(merge_color_seven)
color_attribute_branch = model_color.get_layer('batch_normalization_71').name('batch_normalization_71_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_71').name('activation_71_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('conv2d_72').name('conv2d_72_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('batch_normalization_72').name('batch_normalization_72_xyz')(color_attribute_branch)
color_attribute_branch = model_color.get_layer('activation_72').name('activation_72_xyz')(color_attribute_branch)

color_attribute_branch_two = model_color.get_layer('max_pooling2d_4').name('max_pooling2d_4_xyz')(merge_color_seven)

x = [color_attribute,color_attribute_branch,color_attribute_branch_two]

merge_color_eight = model_color.get_layer('mixed8').name('mixed8_xyz')(x)

############################################################ mixed 9 #######################################################

color_attribute = model_color.get_layer('conv2d_81').name('conv2d_81_xyz')(merge_color_eight)
color_attribute = model_color.get_layer('batch_normalization_81').name('batch_normalization_81_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_81').name('activation_81_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_82').name('conv2d_82_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_82').name('batch_normalization_82_xyz')(color_attribute)

color_attribute_branch_activation = model_color.get_layer('activation_82').name('activation_82_xyz')(color_attribute) # connect to conv_2d_84

color_attribute_first_activation = model_color.get_layer('conv2d_83').name('conv2d_83_xyz')(color_attribute_branch_activation)
color_attribute_first_activation = model_color.get_layer('batch_normalization_83').name('batch_normalization_83_xyz')(color_attribute_first_activation)
color_attribute_first_activation = model_color.get_layer('activation_83').name('activation_83_xyz')(color_attribute_first_activation)
color_attribute_activation = model_color.get_layer('conv2d_84').name('conv2d_84_xyz')(color_attribute_branch_activation)
color_attribute_activation = model_color.get_layer('batch_normalization_84').name('batch_normalization_84_xyz')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('activation_84').name('activation_84_xyz')(color_attribute_activation)

x = [color_attribute_first_activation,color_attribute_activation]

merge_color_one = model_color.get_layer('concatenate_1').name('concatenate_1_xyz')(x)

color_attribute_branch = model_color.get_layer('conv2d_78').name('conv2d_78_xyz')(merge_color_eight)
color_attribute_branch = model_color.get_layer('batch_normalization_78').name('batch_normalization_78_xyz')(color_attribute_branch)
color_attribute_branch_one = model_color.get_layer('activation_78').name('activation_78_xyz')(color_attribute_branch)

color_attribute_branch_sub_branch1 = model_color.get_layer('conv2d_79').name('conv2d_79_xyz')(color_attribute_branch_one)
color_attribute_branch_sub_branch1 = model_color.get_layer('batch_normalization_79').name('batch_normalization_79_xyz')(color_attribute_branch_sub_branch1)
color_attribute_branch_sub_branch1 = model_color.get_layer('activation_79').name('activation_79_xyz')(color_attribute_branch_sub_branch1)

color_attribute_branch_sub_branch2 = model_color.get_layer('conv2d_80').name('conv2d_80_xyz')(color_attribute_branch_one)
color_attribute_branch_sub_branch2 = model_color.get_layer('batch_normalization_80').name('batch_normalization_80_xyz')(color_attribute_branch_sub_branch2)
color_attribute_branch_sub_branch2 = model_color.get_layer('activation_80').name('activation_80_xyz')(color_attribute_branch_sub_branch2)

x = [color_attribute_branch_sub_branch1,color_attribute_branch_sub_branch2]

merge_color_two = model_color.get_layer('mixed9_0').name('mixed9_0_xyz')(x)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_8').name('average_pooling2d_8_xyz')(merge_color_eight)
color_attribute_branch_two = model_color.get_layer('conv2d_85').name('conv2d_85_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_85').name('batch_normalization_85_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_85').name('activation_85_xyz')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_77').name('conv2d_77_xyz')(merge_color_eight)
color_attribute_branch_three = model_color.get_layer('batch_normalization_77').name('batch_normalization_77_xyz')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_77').name('activation_77_xyz')(color_attribute_branch_three)

x = [merge_color_one,merge_color_two,color_attribute_branch_two,color_attribute_branch_three]

merge_color_nine = model_color.get_layer('mixed9').name('mixed9_xyz')(x)

############################################################ Merge 10 ##################################################

color_attribute = model_color.get_layer('conv2d_90').name('conv2d_90_xyz')(merge_color_nine)
color_attribute = model_color.get_layer('batch_normalization_90').name('batch_normalization_90_xyz')(color_attribute)
color_attribute = model_color.get_layer('activation_90').name('activation_90_xyz')(color_attribute)
color_attribute = model_color.get_layer('conv2d_91').name('conv2d_91_xyz')(color_attribute)
color_attribute = model_color.get_layer('batch_normalization_91').name('batch_normalization_91_xyz')(color_attribute)

color_attribute_branch_activation = model_color.get_layer('activation_91').name('activation_91_xyz')(color_attribute) # connect to conv_2d_84

color_attribute = model_color.get_layer('conv2d_92').name('conv2d_92_xyz')(color_attribute_branch_activation)
color_attribute = model_color.get_layer('batch_normalization_92').name('batch_normalization_92_xyz')(color_attribute_branch_activation)
color_attribute = model_color.get_layer('activation_92').name('activation_92_xyz')(color_attribute_branch_activation)

color_attribute_activation = model_color.get_layer('conv2d_93').name('conv2d_93_xyz')(color_attribute_branch_activation)
color_attribute_activation = model_color.get_layer('batch_normalization_93').name('batch_normalization_93_xyz')(color_attribute_activation)
color_attribute_activation = model_color.get_layer('activation_93').name('activation_93_xyz')(color_attribute_activation)

x = [color_attribute_branch_activation,color_attribute_activation]

merge_color_one = model_color.get_layer('concatenate_2').name('concatenate_2_xyz')(x)

color_attribute_branch = model_color.get_layer('conv2d_87').name('conv2d_87_xyz')(merge_color_nine)
color_attribute_branch = model_color.get_layer('batch_normalization_87').name('batch_normalization_87_xyz')(color_attribute_branch)
color_attribute_branch_one = model_color.get_layer('activation_87').name('activation_87_xyz')(color_attribute_branch)

color_attribute_branch_sub_branch1 = model_color.get_layer('conv2d_88').name('conv2d_88_xyz')(color_attribute_branch_one)
color_attribute_branch_sub_branch1 = model_color.get_layer('batch_normalization_88').name('batch_normalization_88_xyz')(color_attribute_branch_sub_branch1)
color_attribute_branch_sub_branch1 = model_color.get_layer('activation_88').name('activation_88_xyz')(color_attribute_branch_sub_branch1)

color_attribute_branch_sub_branch2 = model_color.get_layer('conv2d_89').name('conv2d_89_xyz')(color_attribute_branch_one)
color_attribute_branch_sub_branch2 = model_color.get_layer('batch_normalization_89').name('batch_normalization_89_xyz')(color_attribute_branch_sub_branch2)
color_attribute_branch_sub_branch2 = model_color.get_layer('activation_89').name('activation_89_xyz')(color_attribute_branch_sub_branch2)

x = [color_attribute_branch_sub_branch1,color_attribute_branch_sub_branch2]

merge_color_two = model_color.get_layer('mixed9_1').name('mixed9_1_xyz')(x)

color_attribute_branch_two = model_color.get_layer('average_pooling2d_9').name('average_pooling2d_9_xyz')(merge_color_nine)
color_attribute_branch_two = model_color.get_layer('conv2d_94').name('conv2d_94_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('batch_normalization_94').name('batch_normalization_94_xyz')(color_attribute_branch_two)
color_attribute_branch_two = model_color.get_layer('activation_94').name('activation_94_xyz')(color_attribute_branch_two)

color_attribute_branch_three = model_color.get_layer('conv2d_86').name('conv2d_86_xyz')(merge_color_nine)
color_attribute_branch_three = model_color.get_layer('batch_normalization_86').name('batch_normalization_86_xyz')(color_attribute_branch_three)
color_attribute_branch_three = model_color.get_layer('activation_86').name('activation_86_xyz')(color_attribute_branch_three)

x = [merge_color_one,merge_color_two,color_attribute_branch_two,color_attribute_branch_three]

merge_color_ten = model_color.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

color_attribute = model_color.get_layer('global_average_pooling2d_1').name('global_average_pooling2d_1_xyz')(merge_color_ten)
color_attribute = model_color.get_layer('dropout_1').name('dropout_1_xyz')(color_attribute)
color_attribute = model_color.get_layer('attribute_color').name('attribute_color_xyz')(color_attribute)
predictions_color = model_color.get_layer('predictions_color').name('predictions_color_xyz')(color_attribute)



##########################################################################################################################
################################################## Model for gender #####################################################
##########################################################################################################################

print("Preparing gender Model...")

gender_attribute = model_pattern.get_layer('mixed5').output
gender_attribute = model_gender.get_layer('conv2d_55').name('conv2d_55_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_55').name('batch_normalization_55_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_55').name('activation_55_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_56').name('conv2d_56_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_56').name('batch_normalization_56_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_56').name('activation_56_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_57').name('conv2d_57_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_57').name('batch_normalization_57_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_57').name('activation_57_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_58').name('conv2d_58_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_58').name('batch_normalization_58_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_58').name('activation_58_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_59').name('conv2d_59_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_59').name('batch_normalization_59_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_59').name('activation_59_abc')(gender_attribute)


gender_attribute_branch = model_pattern.get_layer('mixed5').output
gender_attribute_branch = model_gender.get_layer('conv2d_52').name('conv2d_52_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_52').name('batch_normalization_52_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_52').name('activation_52_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_53').name('conv2d_53_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_53').name('batch_normalization_53_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_53').name('activation_53_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_54').name('conv2d_54_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_54').name('batch_normalization_54_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_54').name('activation_54_abc')(gender_attribute_branch)


gender_attribute_branch_two = model_pattern.get_layer('mixed5').output
gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_6').name('average_pooling2d_6_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('conv2d_60').name('conv2d_60_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_60').name('batch_normalization_60_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_60').name('activation_60_abc')(gender_attribute_branch_two)

gender_attribute_branch_three = model_pattern.get_layer('mixed5').output
gender_attribute_branch_three = model_gender.get_layer('conv2d_51').name('conv2d_51_abc')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_51').name('batch_normalization_51_abc')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_51').name('activation_51_abc')(gender_attribute_branch_three)

# Merge layers
x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_six = model_gender.get_layer('mixed6')(x)

########################################################## Merge 7 ###########################################################


gender_attribute = model_gender.get_layer('conv2d_65').name('conv2d_65_abc')(merge_gender_six)
gender_attribute = model_gender.get_layer('batch_normalization_65').name('batch_normalization_65_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_65').name('activation_65_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_66').name('conv2d_66_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_66').name('batch_normalization_66_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_66').name('activation_66_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_67').name('conv2d_67_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_67').name('batch_normalization_67_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_67').name('activation_67_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_68').name('conv2d_68_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_68').name('batch_normalization_68_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_68').name('activation_68_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_69').name('conv2d_69_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_69').name('batch_normalization_69_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_69').name('activation_69_abc')(gender_attribute)


gender_attribute_branch = model_gender.get_layer('conv2d_62').name('conv2d_62_abc')(merge_gender_six)
gender_attribute_branch = model_gender.get_layer('batch_normalization_62').name('batch_normalization_62_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_62').name('activation_62_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_63').name('conv2d_63_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_63').name('batch_normalization_63_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_63').name('activation_63_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_64').name('conv2d_64_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_64').name('batch_normalization_64_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_64').name('activation_64_abc')(gender_attribute_branch)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_7').name('average_pooling2d_7_abc')(merge_gender_six)
gender_attribute_branch_two = model_gender.get_layer('conv2d_70').name('conv2d_70_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_70').name('batch_normalization_70_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_70').name('activation_70_abc')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_61').name('conv2d_61_abc')(merge_gender_six)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_61').name('batch_normalization_61_abc')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_61').name('activation_61_abc')(gender_attribute_branch_three)

# Merge layers
x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_seven = model_gender.get_layer('mixed7')(x)

############################################################# mixed 8 ###################################################

gender_attribute = model_gender.get_layer('conv2d_73').name('conv2d_73_abc')(merge_gender_seven)
gender_attribute = model_gender.get_layer('batch_normalization_73').name('batch_normalization_73_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_73').name('activation_73_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_74').name('conv2d_74_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_74').name('batch_normalization_74_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_74').name('activation_74_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_75').name('conv2d_75_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_75').name('batch_normalization_75_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_75').name('activation_75_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_76').name('conv2d_76_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_76').name('batch_normalization_76_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_76').name('activation_76_abc')(gender_attribute)

gender_attribute_branch = model_gender.get_layer('conv2d_71').name('conv2d_71_abc')(merge_gender_seven)
gender_attribute_branch = model_gender.get_layer('batch_normalization_71').name('batch_normalization_71_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_71').name('activation_71_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('conv2d_72').name('conv2d_72_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('batch_normalization_72').name('batch_normalization_72_abc')(gender_attribute_branch)
gender_attribute_branch = model_gender.get_layer('activation_72').name('activation_72_abc')(gender_attribute_branch)

gender_attribute_branch_two = model_gender.get_layer('max_pooling2d_4').name('max_pooling2d_4_abc')(merge_gender_seven)

x = [gender_attribute,gender_attribute_branch,gender_attribute_branch_two]

merge_gender_eight = model_gender.get_layer('mixed8').name('mixed8_abc')(x)

############################################################ mixed 9 #######################################################

gender_attribute = model_gender.get_layer('conv2d_81').name('conv2d_81_abc')(merge_gender_eight)
gender_attribute = model_gender.get_layer('batch_normalization_81').name('batch_normalization_81_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_81').name('activation_81_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_82').name('conv2d_82_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_82').name('batch_normalization_82_abc')(gender_attribute)

gender_attribute_branch_activation = model_gender.get_layer('activation_82').name('activation_82_abc')(gender_attribute) # connect to conv_2d_84

gender_attribute_first_activation = model_gender.get_layer('conv2d_83').name('conv2d_83_abc')(gender_attribute_branch_activation)
gender_attribute_first_activation = model_gender.get_layer('batch_normalization_83').name('batch_normalization_83_abc')(gender_attribute_first_activation)
gender_attribute_first_activation = model_gender.get_layer('activation_83').name('activation_83_abc')(gender_attribute_first_activation)
gender_attribute_activation = model_gender.get_layer('conv2d_84').name('conv2d_84_abc')(gender_attribute_branch_activation)
gender_attribute_activation = model_gender.get_layer('batch_normalization_84').name('batch_normalization_84_abc')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('activation_84').name('activation_84_abc')(gender_attribute_activation)

x = [gender_attribute_first_activation,gender_attribute_activation]

merge_gender_one = model_gender.get_layer('concatenate_1').name('concatenate_1_abc')(x)

gender_attribute_branch = model_gender.get_layer('conv2d_78').name('conv2d_78_abc')(merge_gender_eight)
gender_attribute_branch = model_gender.get_layer('batch_normalization_78').name('batch_normalization_78_abc')(gender_attribute_branch)
gender_attribute_branch_one = model_gender.get_layer('activation_78').name('activation_78_abc')(gender_attribute_branch)

gender_attribute_branch_sub_branch1 = model_gender.get_layer('conv2d_79').name('conv2d_79_abc')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('batch_normalization_79').name('batch_normalization_79_abc')(gender_attribute_branch_sub_branch1)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('activation_79').name('activation_79_abc')(gender_attribute_branch_sub_branch1)

gender_attribute_branch_sub_branch2 = model_gender.get_layer('conv2d_80').name('conv2d_80_abc')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('batch_normalization_80').name('batch_normalization_80_abc')(gender_attribute_branch_sub_branch2)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('activation_80').name('activation_80_abc')(gender_attribute_branch_sub_branch2)

x = [gender_attribute_branch_sub_branch1,gender_attribute_branch_sub_branch2]

merge_gender_two = model_gender.get_layer('mixed9_0').name('mixed9_0_abc')(x)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_8').name('average_pooling2d_8_abc')(merge_gender_eight)
gender_attribute_branch_two = model_gender.get_layer('conv2d_85').name('conv2d_85_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_85').name('batch_normalization_85_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_85').name('activation_85_abc')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_77').name('conv2d_77_abc')(merge_gender_eight)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_77').name('batch_normalization_77_abc')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_77').name('activation_77_abc')(gender_attribute_branch_three)

x = [merge_gender_one,merge_gender_two,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_nine = model_gender.get_layer('mixed9').name('mixed9_abc')(x)

############################################################ Merge 10 ##################################################

gender_attribute = model_gender.get_layer('conv2d_90').name('conv2d_90_abc')(merge_gender_nine)
gender_attribute = model_gender.get_layer('batch_normalization_90').name('batch_normalization_90_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('activation_90').name('activation_90_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('conv2d_91').name('conv2d_91_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('batch_normalization_91').name('batch_normalization_91_abc')(gender_attribute)

gender_attribute_branch_activation = model_gender.get_layer('activation_91').name('activation_91_abc')(gender_attribute) # connect to conv_2d_84

gender_attribute = model_gender.get_layer('conv2d_92').name('conv2d_92_abc')(gender_attribute_branch_activation)
gender_attribute = model_gender.get_layer('batch_normalization_92').name('batch_normalization_92_abc')(gender_attribute_branch_activation)
gender_attribute = model_gender.get_layer('activation_92').name('activation_92_abc')(gender_attribute_branch_activation)

gender_attribute_activation = model_gender.get_layer('conv2d_93').name('conv2d_93_abc')(gender_attribute_branch_activation)
gender_attribute_activation = model_gender.get_layer('batch_normalization_93').name('batch_normalization_93_abc')(gender_attribute_activation)
gender_attribute_activation = model_gender.get_layer('activation_93').name('activation_93_abc')(gender_attribute_activation)

x = [gender_attribute_branch_activation,gender_attribute_activation]

merge_gender_one = model_gender.get_layer('concatenate_2').name('concatenate_2_abc')(x)

gender_attribute_branch = model_gender.get_layer('conv2d_87').name('conv2d_87_abc')(merge_gender_nine)
gender_attribute_branch = model_gender.get_layer('batch_normalization_87').name('batch_normalization_87_abc')(gender_attribute_branch)
gender_attribute_branch_one = model_gender.get_layer('activation_87').name('activation_87_abc')(gender_attribute_branch)

gender_attribute_branch_sub_branch1 = model_gender.get_layer('conv2d_88').name('conv2d_88_abc')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('batch_normalization_88').name('batch_normalization_88_abc')(gender_attribute_branch_sub_branch1)
gender_attribute_branch_sub_branch1 = model_gender.get_layer('activation_88').name('activation_88_abc')(gender_attribute_branch_sub_branch1)

gender_attribute_branch_sub_branch2 = model_gender.get_layer('conv2d_89').name('conv2d_89_abc')(gender_attribute_branch_one)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('batch_normalization_89').name('batch_normalization_89_abc')(gender_attribute_branch_sub_branch2)
gender_attribute_branch_sub_branch2 = model_gender.get_layer('activation_89').name('activation_89_abc')(gender_attribute_branch_sub_branch2)

x = [gender_attribute_branch_sub_branch1,gender_attribute_branch_sub_branch2]

merge_gender_two = model_gender.get_layer('mixed9_1').name('mixed9_1_abc')(x)

gender_attribute_branch_two = model_gender.get_layer('average_pooling2d_9').name('average_pooling2d_9_abc')(merge_gender_nine)
gender_attribute_branch_two = model_gender.get_layer('conv2d_94').name('conv2d_94_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('batch_normalization_94').name('batch_normalization_94_abc')(gender_attribute_branch_two)
gender_attribute_branch_two = model_gender.get_layer('activation_94').name('activation_94_abc')(gender_attribute_branch_two)

gender_attribute_branch_three = model_gender.get_layer('conv2d_86').name('conv2d_86_abc')(merge_gender_nine)
gender_attribute_branch_three = model_gender.get_layer('batch_normalization_86').name('batch_normalization_86_abc')(gender_attribute_branch_three)
gender_attribute_branch_three = model_gender.get_layer('activation_86').name('activation_86_abc')(gender_attribute_branch_three)

x = [merge_gender_one,merge_gender_two,gender_attribute_branch_two,gender_attribute_branch_three]

merge_gender_ten = model_gender.get_layer('mixed10')(x)

#################################################### Final Layers ########################################################

gender_attribute = model_gender.get_layer('global_average_pooling2d_1').name('global_average_pooling2d_1_abc')(merge_gender_ten)
gender_attribute = model_gender.get_layer('dropout_1').name('dropout_1_abc')(gender_attribute)
gender_attribute = model_gender.get_layer('attribute_gender').name('attribute_gender_abc')(gender_attribute)
predictions_gender = model_gender.get_layer('predictions_gender').name('predictions_gender_abc')(gender_attribute)

final_model = Model(inputs = model_pattern.input, outputs= [predictions_pattern,predictions_color,predictions_gender])
final_model.save("../models/final_model.h5")
print("Model Created.")