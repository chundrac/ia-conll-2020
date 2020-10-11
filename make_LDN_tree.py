import sys
import os
import numpy as np
from util import *

import editdistance
from collections import defaultdict

seed = 0
k = 'all'

lang_raw,input_raw,output_raw,POS_raw,gloss_raw,langs,input_segs,output_segs,POS_segs,X,Y,J,S,T_x,T_y,L,N,lang_id,enc_in,dec_in,POS_in,gloss_in,dec_out = generate_data(seed,k)

lang_raw = lang_raw['train']
input_raw = [''.join(w) for w in input_raw['train']]
output_raw = [''.join(w) for w in output_raw['train']]

reflex_dict = {w:defaultdict(list) for w in sorted(set(input_raw))}
for i in range(N):
    reflex_dict[input_raw[i]][lang_raw[i]].append(output_raw[i])

distmat = np.zeros([L,L])
for i in range(L-1):
    for j in range(i+1,L):
        dists = []
        lang1 = langs[i]
        lang2 = langs[j]
        for etymon in reflex_dict.keys():
            if lang1 in reflex_dict[etymon].keys() and lang2 in reflex_dict[etymon].keys():
                subdists = []
                for w1 in reflex_dict[etymon][lang1]:
                    for w2 in reflex_dict[etymon][lang2]:
                        subdists.append(editdistance.distance(w1,w2)/max([len(w1),len(w2)]))
                dists.append(np.mean(subdists))
        distmat[i,j] = np.mean(dists)
        distmat[j,i] = np.mean(dists)

f = open('trees/LDN_dist.tsv','w')
print('\t'.join(['']+langs),file=f)
for i,l in enumerate(langs):
    print('\t'.join([l]+[str(s) for s in distmat[i,]]),file=f)

f.close()
    
    
    