import pandas as pd
import numpy as np
import os
import random


data = pd.read_csv("../data/category_data.csv")

c_s_dict={}
l_id=list(data['id'])
l_label=list(data['label'])

for i in range(len(l_id)):
   temp = l_id[i].split('/')
   try:
       if (temp[3] not in c_s_dict[l_label[i]]):
           c_s_dict[l_label[i]].append(temp[3])

       else:
           pass
   except:
       c_s_dict[l_label[i]]=[temp[3]]

dir='../data/img'
sub_dir= list(os.walk(dir))[0][1]

import glob
img_dict={}
for i in sub_dir:
       temp=glob.glob(dir+'/'+i+"/*.jpg")
       random.shuffle(temp)

       img_dict[i]=temp

thresold = 6000
final_data_dict = {}

for i in c_s_dict.keys():
   temp_thresold = int(thresold/(len(c_s_dict[i])))
   for j in c_s_dict[i]:
       if (len(img_dict[j])>=temp_thresold):
           for k in range (temp_thresold):
               try:
                   final_data_dict[i].append(img_dict[j][k])
               except:
                   final_data_dict[i] = [img_dict[j][k]]
       else:
           try:
               final_data_dict[i]=final_data_dict[i]+list(img_dict[j])
           except:
               final_data_dict[i]= list(img_dict[j])

import csv

# with open('../data/category_data001.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(('id', 'label'))
#     for key, value in final_data_dict.items():
#         for i in value:
#             writer.writerow([i,key])


img=[]
label=[]
for i in final_data_dict.keys():
  for j in final_data_dict[i]:
    img.append(j)
    label.append(i)
dataframe = pd.DataFrame({'id':img,'label':label})
dataframe.to_csv("../data/category_data001.csv",index = False)
