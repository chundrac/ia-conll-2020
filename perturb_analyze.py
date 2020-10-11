import editdistance
from collections import defaultdict
import numpy as np

#batch = 0

modes = ['langPOS','langPOSsem','langPOSsemetym']


dists = {}
for mode in modes:
  for batch in range(8):
    dists[mode] = defaultdict(list)
    text = [l.strip().split('\t') for l in open('decoded_0_{}_{}.tsv'.format(batch,mode),'r')]
    text0 = [l.strip().split('\t') for l in open('decoded_perturbed_0_{}_{}.tsv'.format(batch,mode),'r')]
    for i in range(len(text)):
        POS = text0[i][1]
        x = text[i][3]
        y = text0[i][4]
        dist = editdistance.distance(x,y)/max([len(x),len(y)])
        dists[mode][POS].append(dist)


for k in dists.keys():
    stats.mannwhitneyu(dists[k]['N'],dists[k]['V'])
    for key in dists[k].keys():
        print(k,key,np.mean(dists[k][key]))
        
    


match = {}
for mode in modes:
    match[mode] = defaultdict(list)
    text = [l.strip().split('\t') for l in open('decoded_0_{}_{}.tsv'.format(batch,mode),'r')]
    text0 = [l.strip().split('\t') for l in open('decoded_perturbed_0_{}_{}.tsv'.format(batch,mode),'r')]
    for i in range(len(text)):
        POS = text0[i][1]
        m = 0
        x = text[i][3][1:-1]
        y = text0[i][4][1:-1]
        for j in range(min([len(x),len(y)])):
            if x[j] == y[j]:
                m += 1
            else:
                break
        dist = m/np.mean([len(x),len(y)])
        #dist = np.mean([len(x)-m,len(y)-m])
        print(x,y,x[m:],y[m:],dist)
        match[mode][POS].append(dist)


for k in match.keys():
    for key in match[k].keys():
        print(k,key,np.mean(match[k][key]))