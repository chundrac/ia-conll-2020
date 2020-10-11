import unicodedata
import numpy as np
from collections import defaultdict

collated = [l.strip().split('\t') for l in open('collated_output.tsv','r')]

true = [list(unicodedata.normalize('NFC',l[2])) for l in collated]

L = [list(unicodedata.normalize('NFC',l[3])) for l in collated]

LP = [list(unicodedata.normalize('NFC',l[5])) for l in collated]

LPS = [list(unicodedata.normalize('NFC',l[7])) for l in collated]

LPSE = [list(unicodedata.normalize('NFC',l[9])) for l in collated]

B = [list(unicodedata.normalize('NFC',l[11])) for l in collated]

segs = sorted(set([s for w in true+L+LP+LPS+LPSE+B for s in w]))

def init_sim():
    S = {}
    for s in segs:
        for t in segs:
            if s == t:
                S[(s,t)] = 1
            else:
                S[(s,t)] = -1
    return(S)



def gensim(pmi):
    S = {}
    for s in segs:
        for t in segs:
            S[(s,t)] = pmi[(s,t)]
    return(S)


def NW(S,a,b): #needleman wunsch
    d = -5
    m=len(a)
    n=len(b)
    F = np.zeros([m+1,n+1])
#    print F
    for i in range(0,m+1):
        F[i,0] = int(d*i)
    for j in range(0,n+1):
        F[0,j] = int(d*j)
#    print F
    for i in range(1,m+1):
        for j in range(1,n+1):
#            print i,j
            match = F[i-1,j-1] + S[(a[i-1],b[j-1])]
            delete = F[i-1,j] + d
            insert = F[i,j-1] + d
#            print max(match,delete,insert)
            F[i,j] = max(match,delete,insert)
#    print F
    alignA = []
    alignB = []
    i = m
    j = n
    while i > 0 or j > 0:
#        print alignA,alignB
        if i > 0 and j > 0 and F[i,j] == F[i-1,j-1] + S[(a[i-1],b[j-1])]:
            alignA.insert(0,a[i-1])
            alignB.insert(0,b[j-1])
            i,j = i-1,j-1
        elif i > 0 and F[i,j] == F[i-1,j] + d:
            alignA.insert(0,a[i-1])
            alignB.insert(0,'-')
            i = i-1
        else:
            alignA.insert(0,'-')
            alignB.insert(0,b[j-1])
            j = j-1
    return(tuple(alignA),tuple(alignB))

def getPMI(etyma,reflexes):
  similarities = []
  #S = init_sim()
  for t in range(1000):
    align_counts = []
    for i in range(len(etyma)):
        #print(etyma[i])
        aligned = NW(S,etyma[i],reflexes[i])
        for i in range(len(aligned[0])):
            align_counts.append((aligned[0][i],aligned[1][i]))
    bi_counts = defaultdict(int)
    etym_counts = defaultdict(int)
    ref_counts = defaultdict(int)
    for x in segs:
        for y in segs:
            bi_counts[(x,y)] += .001
            etym_counts[x] += .001
            ref_counts[y] += .001
    for e in align_counts:
        if '-' not in e:
            bi_counts[e] += 1
            etym_counts[e[0]] += 1
            ref_counts[e[1]] += 1
    pmi = defaultdict(float)
    for k in bi_counts.keys():
        pxy = bi_counts[k]/sum(bi_counts.values())
        px = etym_counts[k[0]]/sum(etym_counts.values())
        py = ref_counts[k[1]]/sum(ref_counts.values())
        pmi[k] = np.log(pxy/(px*py))
    similarities.append(S)
    S = gensim(pmi)
    print(t)
    if len(similarities) >= 4 and sum(np.array(list(similarities[-1].values()))-np.array(list(similarities[-2].values())))==0:
        break
  return(S)


S = getPMI(true,L)
L_pairs = []
for i in range(len(true)):
    L_pairs.append(NW(S,true[i],L[i]))


S = getPMI(true,LP)
LP_pairs = []
for i in range(len(true)):
    LP_pairs.append(NW(S,true[i],LP[i]))


S = getPMI(true,LPS)
LPS_pairs = []
for i in range(len(true)):
    LPS_pairs.append(NW(S,true[i],LPS[i]))


S = getPMI(true,LPSE)
LPSE_pairs = []
for i in range(len(true)):
    LPSE_pairs.append(NW(S,true[i],LPSE[i]))


S = getPMI(true,B)
B_pairs = []
for i in range(len(true)):
    B_pairs.append(NW(S,true[i],B[i]))


pairs = {}
pairs['L'] = L_pairs
pairs['LP'] = LP_pairs
pairs['LPS'] = LPS_pairs
pairs['LPSE'] = LPSE_pairs
pairs['B'] = B_pairs

vowels = ['a', 'e', 'i', 'o', 'u', 'à', 'á', 'â', 'ã', 'ä', 'å', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü', 'ā', 'ă', 'ē', 'ĕ', 'ě', 'ĩ', 'ī', 'ĭ', 'ľ', 'ō', 'ŏ', 'ũ', 'ū', 'ŭ', 'ǒ', 'ǖ', 'ǧ', 'ȧ', 'ȫ', 'ȯ', 'ȳ', 'ɔ', 'ə', 'ʻ', '˅', 'ˊ', 'ˋ', '˘', '˜', '̀', '́', '̂', '̃', '̄', '̆', '̇', '̈', '̊', '̌', '̠', '̣', '̤', '̥', '̦', '̬', '̭', '̮', '̯', '̱', 'ḕ', 'ḗ', 'ṍ', 'ṓ', 'ṳ', 'ṷ', 'ạ', 'ắ', 'ặ', 'ẹ', 'ẽ', 'ễ', 'ị', 'ọ', 'ỗ', 'ụ', 'ỹ']

edits = {k:defaultdict(int) for k in pairs.keys()}
for k in pairs.keys():
    for i in range(len(pairs[k])):
        x,y = pairs[k][i]
        for j in range(len(x)):
            #edits[k][(x[j],y[j])] += 1
          if x[j] != y[j]:
            if x[j] == '-':
                edits[k]['insertion'] += 1
            elif y[j] == '-':
                edits[k]['deletion'] += 1
            elif x[j] in vowels:
                edits[k]['vowel'] += 1
            else:
                print(x[j],y[j])
                edits[k]['consonant'] += 1



for k in edits:
  print(k,'&',end=' ')
  for key in ['insertion','deletion','vowel','consonant']:
    print("{:.2f}".format((edits[k][key]/sum(edits[k].values()))),'&',end=' ')
  print('\\\\')




mismatch = defaultdict(int)
for k in pairs.keys():
    for i in range(len(pairs[k])):
        x,y = pairs[k][i]
        for j in range(len(x)):
          if x[j] != y[j]:
              mismatch[(x[j],y[j])] += 1



edits = {k:defaultdict(int) for k in pairs.keys()}
for k in pairs.keys():
    for i in range(len(pairs[k])):
        x,y = pairs[k][i]
        for j in range(len(x)):
            #edits[k][(x[j],y[j])] += 1
          if x[j] != y[j]:
            if x[j] == '-':
                edits[k]['insertion'] += 1
            elif y[j] == '-':
                edits[k]['deletion'] += 1
                if k != 'B':
                    print(''.join(pairs[k][i][0]),''.join(pairs[k][i][1]),''.join(pairs['B'][i][0]),''.join(pairs['B'][i][1]))
            elif x[j] in vowels:
                edits[k]['vowel'] += 1
            else:
                #print(x[j],y[j])
                edits[k]['consonant'] += 1