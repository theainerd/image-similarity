import numpy as np
import pandas as pd
import pickle
from annoy import AnnoyIndex

import pickle

import glob
pickle_path = "../data/pickle/" #Same Path as out_path i.e. where the pickle files are

data_p_files=[]
for name in glob.glob(pickle_path + "data_*.pkl"):
    data_p_files.append(name)

data = pd.DataFrame([])
for i in range(len(data_p_files)):
    data = data.append(pd.read_pickle(data_p_files[i]),ignore_index=True)

print(traindf)

def _index_and_search(k,query_vector):
   xb = []                # array to be indexed
   xb = list(traindf['vector'])

   d = 1024
   nb = len(xb)                 # database size
   nq = 1                      # nb of queries
   annoy_index(d, xb)
   # print("End Indexing"+str(datetime.datetime.now()))

   xq = 0

   try:
       xq = xb.index(query_vector)
       # print("Start Searching"+str(datetime.datetime.now()))
       I = annoy_search(xq, 'annoy_index.ann', k, d) # search results
       # print("End Searching"+str(datetime.datetime.now()))
       # return {"I":I}
       return explain_result(I)
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

print(_index_and_search(5,traindf['vector'][1]))