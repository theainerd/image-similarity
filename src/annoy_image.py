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

def _index_and_search(posts, k, query_id):
    .0xb = []                         # array to be indexed
    data = posts["hits"]["hits"]
    if len(data) == 0:
        return {"message":"no items found for filter"}

    # print("data length in index: "+str(len(data)))
    for i, item in enumerate(list(data)):
        xb.append(item["_source"]["vector"])

    d = VECTOR_DIMENSION
    nb = len(xb)                 # database size
    nq = QUERY_LENGTH                      # nb of queries

    query_vector = [] # vector to be queried
    terms = {"_id":[query_id]}
    filter = [{"terms": terms}]
    l, query_post = es_search._search_single(filter)
    if l == 0:
        return {"message":"query_id not found in database"}
    else:
        # print("append query vector to xb")
        query_vector = query_post["_source"]["vector"]
        xb.append(query_vector)
        data.append(query_post)
    # print("Start Indexing"+str(datetime.datetime.now()))
    annoy_index(d, xb)
    # print("End Indexing"+str(datetime.datetime.now()))

    xq = 0

    if l == 0:
        return {"message":"query_id not found in database"}
    else:
        query_vector = query_post["_source"]["vector"]
        try:
            xq = xb.index(query_vector)
            # print("Start Searching"+str(datetime.datetime.now()))
            I = annoy_search(xq, 'annoy_index.ann', k, d) # search results
            # print("End Searching"+str(datetime.datetime.now()))
            # return {"I":I}
            return explain_result(I, data, query_post)
        except Exception as e:
            raise
            return {"message":"query_id not found in index"}

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
