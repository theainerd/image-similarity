from annoy import AnnoyIndex
intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[
                                     model.get_layer('dropout_5').output
                                     ])

preds = intermediate_layer_model.predict(data)
vector = []
for i, l in enumerate(preds):
    print(type(l))

print(str(preds))
print("vector: "+str(len(preds[0][0])))



t = AnnoyIndex(f)  # Length of feature vector to be indexed
for i, v in enumerate(iterator):  # iterator returns the features
    t.add_item(i, v)  # add item to annoy index

t.build(10) # build model with 10 trees
t.save('annoy_model.ann')  # save model

# u = AnnoyIndex(f)
# u.load('annoy_model.ann') # super fast, will just mmap the file
print(t.get_nns_by_item(0, 10)) # returns top 10 nearest neighbours
