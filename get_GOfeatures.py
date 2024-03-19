import numpy as np
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split
import math
from tensorflow import keras
from stellargraph import globalvar, IndexedArray
import pandas as pd
from stellargraph import StellarGraph
import pickle
#build the GO graph
source = []
target = []
weight = []
with open("go-terms.edgelist", "r") as file:
    for line in file:
        line = line.strip()
        source.append(line.split()[0])
        target.append(line.split()[1])
edges = pd.DataFrame(
            {"source": source, "target": target}
        )
Gs = StellarGraph(edges=edges)

dic=dict(Gs.node_degrees())
with open('go_id_dict', 'rb') as f:
    my_object = pickle.load(f)
new_dict={}
for c in my_object:
    new_dict[my_object[c]]=c[3:]
import csv
def read_tab_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            key = row[0]
            values = row[1:]
            data[key] = values
    return data
#get IC value of every GO term
file_path = 'ic2.ext.tab'
result = read_tab_file(file_path)
dict={}
for c in new_dict:
    try :
        dict[c] = float(result[new_dict[c]][0])
    except:
        dict[c] =0

source = []
target = []
weight = []
with open("go-terms.edgelist", "r") as file:
    for line in file:
        line = line.strip()
        source.append(line.split()[0])
        target.append(line.split()[1])
        s=line.split()[0]
        t=line.split()[1]
        wei = dict[int(s)]*math.sqrt(dic[s])+dict[int(t)]*math.sqrt(dic[t])
        wei = wei/2
        weight.append(wei)
edges = pd.DataFrame(
            {"source": source, "target": target,"weight": weight}
        )
import csv
index=[]
feature_array = []
with open('emb64.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        index.append(row[0])
        feature_array.append(row[1:])

feature_array=np.array(feature_array)
feature_array = feature_array.astype(float)
# As a IndexedArray (no column names):
nodes = IndexedArray(feature_array, index=index)
G = StellarGraph(nodes, edges)
nodes = list(G.nodes())
# print(nodes)
number_of_walks = 10
length = 10
unsupervised_samples = UnsupervisedSampler(G, nodes=nodes, length=length, number_of_walks=number_of_walks)
batch_size = 100
epochs = 5
num_samples = [5, 10]#num_samples [0] indicates the number of samples from the first-layer neighbors, and num_samples [1] indicates the number of samples from the second-layer neighbors.
generator = GraphSAGELinkGenerator(G, batch_size, num_samples,weighted=True)
train_gen = generator.flow(unsupervised_samples)
layer_sizes = [128, 128]
graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.01, normalize="l2")
# Build the model and expose input and output sockets of graphsage, for node pair inputs:
x_inp, x_out = graphsage.in_out_tensors()
prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)
# print(model.summary())
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)
print(model.summary())
history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)

x_inp_src=x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
# print(embedding_model.summary())
from stellargraph.mapper import GraphSAGENodeGenerator
list = G.nodes()
dict={}
for c in list:
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow([c])
    for batch in node_gen:
        node_embeddings = embedding_model.predict([[batch[0][0], batch[0][1], batch[0][2]]], workers=4, verbose=1)
        dict[c]=node_embeddings
    # print(batch[0][1].shape)
# print(dict)
#get vector of every GO term
import pickle
with open('dict_data128.pkl', 'wb') as file:
    pickle.dump(dict, file)


