import os
import csv
import json
import requests
import datetime
import string
import decimal
import re
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
from random import shuffle
import random
import numpy as np
import pandas as pd
import pickle
import sys
from annoy import AnnoyIndex

import env
import control

xb = traindf["vector"].tolist()
query_vector = [] #vector to be queried

xb.append(query_vector)

annoy_index(d, xb)
xq = xb.index(query_vector)
I = annoy_search(xq, 'annoy_index.ann', k, d)
explain_result(I, data, query_post)

def get_value_for_key(key, object):
    if key in object:
        return object[key]
    else:
        return ""

def annoy_index(d, xb):
    t = AnnoyIndex(d)  # Length of item vector that will be indexed
    for i, x in enumerate(xb):
        t.add_item(i, x)

    t.build(20) # 10 trees
    t.save('annoy_index.ann')

def annoy_search(xq, index, k, d):
    u = AnnoyIndex(d)
    u.load(index) # super fast, will just mmap the file
    I = u.get_nns_by_item(xq, k)
    return I

def explain_result(I, data, query_post):
    # print("data length in explain: "+str(len(data)))
    similar_items = []
    for j, nearest_serial in enumerate(I):
        post = data[nearest_serial]
        similar_items.append(post)
        # print("SERIAL: "+str(nearest_serial)+" -->")
        # print("ID: "+str(post["_id"]))
        # print("NAME: "+post["name"])
        # print("DESCRIPTION TEXT "+post["description_text"])
        # try:
        #     print("CLASSIFICATION: "+json.dumps(post["classification"]))
        # except Exception as e:
        #     print("CLASSIFICATION: "+'{"l1":"","l2":"","l3":"","l4":""}')


    return {"message": "items found", "k":len(similar_items), "query_item": query_post, "knn_items": similar_items}
