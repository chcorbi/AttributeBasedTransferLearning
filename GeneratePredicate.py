import os,sys
import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Flatten, merge, Dense, Dropout, Lambda
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from scipy.spatial.distance import hamming
from sklearn.preprocessing import MinMaxScaler

def hamming_distance_scores(att):
    n_classes, n_attributes = att.shape
    scores = np.empty((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            scores[i][j] = hamming(att[i], att[j])
    return scores

def get_cross_corr_scores(att):
    n_classes, n_attributes = att.shape
    scores = np.empty((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            scores[i][j] = np.correlate(att[i], att[j])
    return scores

def random_scores(att):
    n_classes, n_attributes = att.shape
    scores = np.empty((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            scores[i][j] = np.random.random()
    return scores

def create_dataset(scores, include_same_class = False):
    n_classes = scores.shape[0]
    ids = []
    ranks = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j and not include_same_class:
                continue
            ids.append([i, j])
            ranks.append(scores[i][j])
    return np.array(ids), np.array(ranks)

attributes = np.array(pd.read_table('predicate-matrix-binary.txt', delimiter=' ', header=None))
n_classes = attributes.shape[0]

try:
    p_type = str(sys.argv[1])
except IndexError:
    p_type = 'crosscorr'
try:
    if int(sys.argv[2]) > 0:
        include_same_class = True
    else:
        include_same_class = False
except IndexError:
    include_same_class = False

np.random.seed(42)
if p_type == 'hamming':
    scores = hamming_distance_scores(attributes)
    f = open('predicate-matrix-hamming.txt', 'w')
elif p_type == 'crosscorr':
    scores = get_cross_corr_scores(attributes)
    f = open('predicate-matrix-crosscorr.txt', 'w')
else:
    f = open('predicate-matrix-random.txt', 'w')
    scores = random_scores(attributes)

rec_ids, rec_scores = create_dataset(scores, include_same_class)
rec_scl = MinMaxScaler()
rec_scores = rec_scl.fit_transform(rec_scores)

animal1_id_input = Input(shape=[1], name='animal1')
animal2_id_input = Input(shape=[1], name='animal2')

try:
    embedding_size = int(sys.argv[2])
except IndexError:
    embedding_size = 50
emb_layer = Embedding(output_dim=embedding_size, input_dim = n_classes,
                             input_length=1, name='animal_embedding')

animal1_embedding = emb_layer(animal1_id_input)
animal2_embedding = emb_layer(animal2_id_input)

animal1_vecs = Flatten()(animal1_embedding)
animal2_vecs = Flatten()(animal2_embedding)

y = merge([animal1_vecs, animal2_vecs], mode='dot', output_shape=(1,))

model = Model(input=[animal1_id_input, animal2_id_input], output=y)
model.compile(optimizer='adam', loss='mae')

model.fit([rec_ids[:,0], rec_ids[:,1]], rec_scores, batch_size=32, nb_epoch=50, shuffle=True, verbose=0)

embeddings = model.layers[2].get_weights()[0]

scl = MinMaxScaler()
embeddings = scl.fit_transform(embeddings)

s = ''
for i in range(n_classes):
    for j in range(len(embeddings[i])):
        s += str(embeddings[i][j])
        if j != len(embeddings[i]) - 1:
            s += ' '
    if i != n_classes - 1:
        s += '\n'
f.write(s)
