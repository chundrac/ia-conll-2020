from collections import defaultdict
import os

results = defaultdict(list)

modes = ['lang','langPOS','langPOSsem','langPOSsemetym','null']

for mode in modes:
    for i in range(8):
        for l in open('decoded_0_{}_{}.tsv'.format(i,mode)):
            results[mode].append(l.strip().split('\t'))


collated_colnames = ['lang','etymon','true']
for m in modes:
    collated_colnames.append('pred_'+m)
    collated_colnames.append('PER_'+m)


collated = results[modes[0]]
for i in range(len(collated)):
    for m in modes[1:]:
        collated[i] += results[m][i][3:]


print(np.mean([float(l[4]) for l in collated]))
print(np.mean([float(l[6]) for l in collated]))
print(np.mean([float(l[8]) for l in collated]))
print(np.mean([float(l[10]) for l in collated]))
print(np.mean([float(l[12]) for l in collated]))


for i in boundary.keys():
    print(text[i][1],text[i][2],text[i][4],text[i][3][:boundary[i]],text[i][3][boundary[i]:])


stats.wilcoxon([float(l[6]) for l in collated],[float(l[12]) for l in collated])


avgs = {i:np.mean([float(l[i]) for l in collated]) for i in [4,6,8,10,12]}
ordered_keys = sorted(avg.keys(),key=lambda x:avgs[x])


for i,key in enumerate(ordered_keys[:-1]):
    print(collated)
    stats.wilcoxon([float(l[key]) for l in collated],[float(l[ordered_keys[i+1]]) for l in collated])
    


for i in boundary.keys():
    print(text[i][1],text[i][2],text[i][4],text[i][3][:boundary[i]],text[i][3][boundary[i]:])


stats.wilcoxon([float(l[6]) for l in collated],[float(l[12]) for l in collated])


avgs = {i:np.median([float(l[i]) for l in collated]) for i in [4,6,8,10,12]}
ordered_keys = sorted(avgs.keys(),key=lambda x:avgs[x])

WER = {i:len([l for l in collated if float(l[i]) == 0])/len(collated) for i in [4,6,8,10,12]}


for i,key in enumerate(ordered_keys[:-1]):
    print(collated_colnames[key],collated_colnames[ordered_keys[i+1]])
    stats.wilcoxon([float(l[key]) for l in collated],[float(l[ordered_keys[i+1]]) for l in collated],alternative='less')


langs = sorted(set([l[0] for l in collated]))
lang_counts = [[l[0] for l in collated].count(lang) for lang in langs]
lang_accuracy = defaultdict(list)
for i,mode in enumerate(modes):
    j = 4 + (i*2)
    lang_accuracy[mode] = [np.mean([float(l[j]) for l in collated if l[0] == lang]) for lang in langs]


import matplotlib.pyplot as pl    
import matplotlib.colors as mcolors
from tikzplotlib import save as tikz_save


labels = ['L','LP','LPS','LPSE','Baseline']
c = 0
for k in lang_accuracy.keys():
    a,b=np.polyfit(lang_counts,lang_accuracy[k],1)
    plt.plot(lang_counts,lang_accuracy[k],'.',alpha=.4,color=list(mcolors.TABLEAU_COLORS.values())[c])
    plt.plot(lang_counts,a*np.array(lang_counts)+b,color=list(mcolors.TABLEAU_COLORS.values())[c],label=labels[c])
    plt.legend()
    c+=1


labels = ['L','LP','LPS','LPSE','Baseline']
c = 0
for k in lang_accuracy.keys():
    a,b=np.polyfit(np.log(lang_counts),lang_accuracy[k],1)
    plt.plot(np.log(lang_counts),lang_accuracy[k],'.',alpha=.4,color=list(mcolors.TABLEAU_COLORS.values())[c])
    plt.plot(np.log(lang_counts),a*np.array(np.log(lang_counts))+b,color=list(mcolors.TABLEAU_COLORS.values())[c],label=labels[c])
    plt.legend()
    c+=1


tikz_save('lang_PER.tex')



for k in lang_accuracy.keys():
    stats.spearmanr(lang_counts,lang_accuracy[k])
    
    a,b=np.polyfit(np.log(lang_counts),lang_accuracy[k],1)
    plt.plot(np.log(lang_counts),lang_accuracy[k],'.',alpha=.4,color=list(mcolors.TABLEAU_COLORS.values())[c])
    plt.plot(np.log(lang_counts),a*np.array(np.log(lang_counts))+b,color=list(mcolors.TABLEAU_COLORS.values())[c],label=labels[c])
    plt.legend()
    c+=1



random.seed(0)
sampled = random.sample(collated,1000)

for i in range(len(sampled)):
    for k in (4,6,8,10,12):
        sampled[i][k] = sampled[i][k][:3]

f=open('error_forms.tsv','w')
for l in sampled:
    print('\t'.join(l),file=f)


f.close()


f = open('collated_output.tsv','w')
for l in collated:
    print('\t'.join(l),file=f)


f.close()