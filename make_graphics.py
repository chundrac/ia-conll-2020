import numpy as np
from collections import defaultdict
import unicodedata
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

K = 8
text = [unicodedata.normalize('NFD',l).strip('\n').split('\t') for l in open('cdial_wordlist.csv','r')]
glossbert = {l.split('\t')[0]:np.array([float(s) for s in l.strip('\n').split('\t')[1].split()]) for l in open('bert_unique_gloss_PCA.tsv','r')}
exclude = ['ashk1246','kati1270','treg1243','pras1239','waig1243'] #get rid of Nuristani languages
langcount = defaultdict(int)
for l in text:
    langcount[l[0]] += 1

exclude = set(exclude + [k for k in langcount.keys() if langcount[k] < 100])
text = [l+[glossbert[l[3]]] for l in text if l[0] not in exclude]
text = [l for l in text if l[0] not in exclude]
lang_list = defaultdict(list)
for l in text:
    lang_list[l[0]].append(l)

[np.random.shuffle(lang_list[k]) for k in lang_list.keys()]
lang_batches = defaultdict(list)
for key in lang_list.keys():
    breaks = list(range(0,len(lang_list[key]),int(len(lang_list[key])/K)+1))+[len(lang_list[key])]
    for i,b in enumerate(breaks[:-1]):
        lang_batches[key].append(lang_list[key][breaks[i]:breaks[i+1]])      

langs = list(lang_list.keys())

langkey = [l.strip().split('\t') for l in open('lang_key.csv','r')]
langdict={}
for l in langkey:
  langdict[l[3]]=l[2].replace(' ','')+'['+l[3]+']'

langs_ = sorted(langs,key=lambda x:langcount[x])[::-1]

index_ = list(range(len(langs_)))
counts_  = [langcount[l] for l in langs_]
langs_ = [langdict[l] for l in langs_]

plt.clf()
plt.xticks(index_, langs_, rotation='vertical', fontsize=5)
plt.step(index_, counts_)
#figure = plt.gcf()
#figure.set_size_inches(8, 5)
tikz_save('language_counts.tex',axis_height='8cm', axis_width='28cm')

for i,l in enumerate(langs_):
    print(l.split('[')[0]+' & '+l.split('[')[1][:-1]+' & '+str(counts_[i])+'\\\\')