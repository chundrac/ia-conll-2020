import numpy as np
from collections import defaultdict
import unicodedata


#add etymon IDs
def generate_data(seed,k):
    K = 8
    np.random.seed(seed)
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
    train = [l for v in lang_batches.values() for i in range(K) for l in v[i] if i != k]  
    test = [l for v in lang_batches.values() for i in range(K) for l in v[i] if i == k]  
    langs = list(lang_list.keys())  
    input_segs = sorted(set([s for l in text for s in list(l[2])]))
    output_segs = sorted(set([s for l in text for s in ['#']+list(l[1])+['$']]))
    POS_segs = sorted(set([s for l in text for s in list(l[4])]))
    #etyma = [l[2] for l in text]
    lang_raw = defaultdict(list)  
    input_raw = defaultdict(list)  
    output_raw = defaultdict(list)
    POS_raw = defaultdict(list)
    gloss_raw = defaultdict(list)
    #etymon_raw = defaultdict(list)
    lang_raw['train'] = [l[0] for l in train]  
    lang_raw['test'] = [l[0] for l in test]  
    input_raw['train'] = [list(l[2]) for l in train]  
    input_raw['test'] = [list(l[2]) for l in test]  
    output_raw['train'] = [['#']+list(l[1])+['$'] for l in train]  
    output_raw['test'] = [['#']+list(l[1])+['$'] for l in test]  
    POS_raw['train'] = [l[4] for l in train]
    POS_raw['test'] = [l[4] for l in test]
    gloss_raw['train'] = [l[5] for l in train]
    gloss_raw['test'] = [l[5] for l in test]
    #etymon_raw['train'] = [l[2] for l in train]
    #etymon_raw['test'] = [l[2] for l in test]
    gloss_in = np.stack(gloss_raw['train'],0)
    X = len(input_segs)  
    Y = len(output_segs)  
    J = len(POS_segs)
    S = gloss_in.shape[1]
    T_x = max([len(l) for l in input_raw['train']+input_raw['test']])  
    T_y = max([len(l) for l in output_raw['train']+output_raw['test']])
    L = len(langs)
    #M = len(etyma)
    N = len(train)  
    lang_id = np.zeros([N,L],dtype='float32')  
    for i,l in enumerate(lang_raw['train']):  
        lang_id[i,langs.index(l)] = 1.  
    enc_in  = np.zeros([N,T_x,X],dtype='float32')  
    for i,l in enumerate(input_raw['train']):  
        for j,s in enumerate(l):  
            enc_in[i,j,input_segs.index(s)] = 1.  
    dec_in  = np.zeros([N,T_y,Y],dtype='float32')  
    dec_out = np.zeros([N,T_y,Y],dtype='float32')  
    for i,l in enumerate(output_raw['train']):  
        for j,s in enumerate(l):  
            dec_in[i,j,output_segs.index(s)] = 1.  
            if j > 0:
                dec_out[i,j-1,output_segs.index(s)] = 1.
    POS_in = np.zeros([N,J])
    for i,l in enumerate(POS_raw['train']):
        POS_in[i,POS_segs.index(l)] = 1.
    #etymon_id = np.zeros([N,M],dtype='float32')
    #for i,l in enumerate(etymon_raw['train']):  
    #    etymon_id[i,etyma.index(l)] = 1.
    #return(lang_raw,input_raw,output_raw,POS_raw,gloss_raw,etymon_raw,langs,input_segs,output_segs,POS_segs,etyma,X,Y,J,S,T_x,T_y,L,M,N,lang_id,etymon_id,enc_in,dec_in,POS_in,gloss_in,dec_out)
    return(lang_raw,input_raw,output_raw,POS_raw,gloss_raw,langs,input_segs,output_segs,POS_segs,X,Y,J,S,T_x,T_y,L,N,lang_id,enc_in,dec_in,POS_in,gloss_in,dec_out)


def decode_sequence(input_seq,lang_id,POS_seq,sem_seq,langs,output_segs,input_segs,POS_segs,L,X,Y,J,S,T_x,T_y,model,mode):
    lang_id_ = np.zeros([1,L])
    input_seq_ = np.zeros([1,T_x,X])
    POS_seq_ = np.zeros([1,J])
    lang_id_[0,langs.index(lang_id)] = 1.
    for i,s in enumerate(input_seq):
        if s in input_segs:
            input_seq_[0,i,input_segs.index(s)] = 1.
    POS_seq_[0,POS_segs.index(POS_seq)] = 1.
    lang_id = lang_id_
    input_seq = input_seq_
    POS_seq = POS_seq_
    sem_seq = np.expand_dims(sem_seq,0)
    string = ['#']    
    target_seq = np.zeros((1, T_y, Y))
    target_seq[0, 0, output_segs.index('#')] = 1.
    for t in range(T_y-1):
        if mode == 'lang':
            output_tokens = model.predict([lang_id,input_seq,target_seq])
        if mode == 'langPOS':
            output_tokens = model.predict([lang_id,POS_seq,input_seq,target_seq])
        if mode == 'langPOSsem':
            output_tokens = model.predict([lang_id,POS_seq,sem_seq,input_seq,target_seq])
        if mode == 'langPOSsemetym':
            output_tokens = model.predict([lang_id,POS_seq,sem_seq,input_seq,target_seq])
        curr_index = np.argmax(output_tokens[:,t,:])
        target_seq[0, t+1, curr_index] = 1.
        symbol = output_segs[curr_index]
        string.append(symbol)
        if symbol == '$':
            break
    return(string)