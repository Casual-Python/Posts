# https://nlp.stanford.edu/projects/glove/

import numpy as np
def cos_sim(e_w_1,e_w_2):
    return np.dot(e_w_1,e_w_2)\
    /np.linalg.norm(e_w_1)\
    /np.linalg.norm(e_w_2)
  
embeddings_index = {}
with open('glove.6B.300d.txt', encoding="utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

gen_s = np.array(list(embeddings_index.values()))

mean_w_e = gen_s.mean(axis = 0)
cos_sim(embeddings_index['king']-embeddings_index['man'],embeddings_index['queen']-embeddings_index['woman'])
cos_sim(embeddings_index['man'] - mean_w_e ,embeddings_index['woman'] - mean_w_e)
