import json
import os
import sys
import io
import requests
from io import BytesIO
import json
import time
import requests
import datetime
import random
import string
import urllib.parse
import decimal
import uuid
import re
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
import urllib.request

import sys, os, multiprocessing, urllib3, csv
from PIL import Image
from io import BytesIO
from tqdm  import tqdm
import json
import pandas

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

client = urllib3.PoolManager(500)

import pandas as pd
import csv




# with open("data/final_data.csv", 'r') as data_file:
#   reader = csv.reader(data_file, delimiter=',')
#   for row in reader:
#     temp=[x.strip() for x in row[0].split(',')]
#     temp[0]= str(int(temp[0]))
#     data.append(temp)
# print(len(data))
# print(data[0])
# data=data[320000:len(data)]

# def DownloadImage(post):
#  # get URL of the downloaded images
#  out_dir = "data/images/"

#  url = post[1]
#  # url = url[:url.find('?')]
#  # url = url.replace("https:","http:")
#  if not (url.startswith("https:") or url.startswith("http:")):
#    url = "https:"+url
#  key = str(post[0])

#  subdirectory = out_dir
#  filename = subdirectory+key+".jpg"
#  print(filename)

#  try:
#      if not os.path.exists(subdirectory):
#        os.makedirs(subdirectory)
#  except Exception as e:
#      print('Location: %s already exists.' % subdirectory)
#      return


#  if os.path.exists(filename):
#    print('Image %s already exists. Skipping download.' % filename)
#    return

#  try:
#    # global client
#    # response = client.request('GET', url)#, timeout=30)
#    # image_data = response.data
#    response = requests.get(url)
#    image_data = response.content
#  except:
#    print('Warning: Could not download image %s from %s' % (key, url))
#    return

#  try:
#    pil_image = Image.open(BytesIO(image_data))
#  except:
#    print('Warning: Failed to parse image %s %s' % (key,url))
#    return

#  try:
#    pil_image = pil_image.resize((400, 400),Image.ANTIALIAS)
#  except:
#    print('Warning: Failed to resize image %s %s' % (key,url))
#    return

#  try:
#    pil_image_rgb = pil_image.convert('RGB')
#  except:
#    print('Warning: Failed to convert image %s to RGB' % key)
#    return

#  try:
#    pil_image_rgb.save(filename, format='JPEG', quality=90)
#    # print('Success: Proceeding to save image %s' % filename)
#  except:
#    print('Warning: Failed to save image %s' % filename)
#    return


# for batch in range(0,len(data),1000):
#  try:
#     int(batch)
#     pool = ThreadPool(40)
#     list_view_objects = data[batch:batch+1000]
#     print(batch)
#     pool.map(DownloadImage, list_view_objects)
#     pool.close()
#     pool.join()
#      # new_pdp_collection.insert(results)

#  except Exception as e:
#    print('error')