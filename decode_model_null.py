import sys
import os
import numpy as np
from util import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Lambda, dot, RepeatVector, Concatenate
import tensorflow as tf
import tensorflow.keras.backend as K
import editdistance

"""
options:
python3 run_model seed k mode = {'lang','langPOS','langsem','langsemetym'}
"""


assert(len(sys.argv) == 3)
seed = int(sys.argv[1])
k = int(sys.argv[2])
mode = 'null'

#if len(sys.argv) == 4:
#    seed = int(sys.argv[1])
#    k = int(sys.argv[2])
#    mode = sys.argv[3]
#else:
#    seed = 0
#    k = 0
#    mode = 'lang'


lang_raw,input_raw,output_raw,POS_raw,gloss_raw,langs,input_segs,output_segs,POS_segs,X,Y,J,S,T_x,T_y,L,N,lang_id,enc_in,dec_in,POS_in,gloss_in,dec_out = generate_data(seed,k)

latent_dim = 128
embed_dim = 128
hidden_dim = 128
batch_size = 64

lang_id_ = Input((L,))
enc_in_ = Input(shape=(T_x,X))
dec_in_ = Input(shape=(T_y,Y))
enc_mask = tf.expand_dims(tf.reduce_sum(enc_in_,-1),-1)
dec_mask = tf.expand_dims(tf.reduce_sum(dec_in_,-1),-1)
POS_in_ = Input(shape=(J))
sem_in_ = Input(shape=(S))

z_lang = Dense(latent_dim,use_bias=False)(lang_id_)
z_POS = Dense(latent_dim,use_bias=False)(POS_in_)
z_sem = Dense(latent_dim,use_bias=False)(sem_in_)
z_etym = Bidirectional(LSTM(latent_dim,return_sequences=False),'concat')(enc_in_)
z_etym = Dense(latent_dim)(Concatenate()([z_etym,sem_in_]))

embed_lang = Model(inputs = lang_id_,outputs = z_lang)
embed_POS = Model(inputs = POS_in_,outputs = z_POS)
embed_sem = Model(inputs = sem_in_,outputs = z_sem)
embed_etym = Model(inputs = [enc_in_,sem_in_], outputs = z_etym)

embed_lang_in_ = Input((latent_dim,))
embed_POS_in_ = Input((latent_dim,))
embed_sem_in_ = Input((latent_dim,))
embed_etym_in_ = Input((latent_dim,))

embedding = Dense(embed_dim)(enc_in_)

h_enc = Bidirectional(LSTM(hidden_dim, return_sequences=True),'concat')(embedding)*enc_mask
h_dec = LSTM(hidden_dim, return_sequences=True, activation=None)(dec_in_)*dec_mask
#alignment_probs_,emission_probs = monotonic_alignment([h_enc,h_dec,T_x,T_y,Y,hidden_dim])
struc_zeros = K.expand_dims(K.cast(np.triu(np.ones([T_x,T_x])),dtype='float32'),0)
alignment_probs = K.softmax(dot([Dense(hidden_dim)(h_enc),h_dec],axes=-1,normalize=False),-2)
h_enc_rep = K.tile(K.expand_dims(h_enc,-2),[1,1,T_y,1])
h_dec_rep = K.tile(K.expand_dims(h_dec,-3),[1,T_x,1,1])
h_rep = K.concatenate([h_enc_rep,h_dec_rep],-1)

alignment_probs_ = []
for i in range(T_y):
    if i == 0:
        align_prev_curr = tf.gather(alignment_probs,i,axis=-1)
    if i > 0:
        align_prev_curr = tf.einsum('nx,ny->nxy',tf.gather(alignment_probs,i,axis=-1),alignment_probs_[i-1])
        align_prev_curr *= struc_zeros
        align_prev_curr = K.sum(align_prev_curr,1)+1e-6
        align_prev_curr /= K.sum(align_prev_curr,-1,keepdims=True)
    alignment_probs_.append(align_prev_curr)

alignment_probs_ = K.stack(alignment_probs_,-1)
emission_probs = Dense(hidden_dim*3,activation='tanh')(h_rep)
emission_probs = Dense(Y, activation='softmax')(emission_probs)

alignment_probs_ = Lambda(lambda x:x)(alignment_probs_)

alignment_model = Model([enc_in_,dec_in_],alignment_probs_)
alignment_output = alignment_model([enc_in_,dec_in_])
alignment = Model([enc_in_,dec_in_],alignment_output)

alphas = tf.expand_dims(alignment_probs_,-1)*emission_probs
dec_out_ = tf.reduce_sum(alphas,-3)
dec_out_ = Lambda(lambda x:x)(dec_out_)

decoder = Model([enc_in_,dec_in_],dec_out_)
output = decoder([enc_in_,dec_in_])
model = Model(inputs=[enc_in_,dec_in_], outputs=output)
model.compile(optimizer='adam',loss='categorical_crossentropy')

#no_improve = 0
#val_loss_old = 1000
#if 'loss_no_improvement_{}_{}_{}.txt'.format(seed,k,mode) in os.listdir('.'):
#    f = open('loss_no_improvement_{}_{}_{}.txt'.format(seed,k,mode),'r')
#    text = f.read().strip()
#    no_improve,val_loss_old = int(text.split()[0]),float(text.split()[1])
#    f.close()

if 'model_{}_{}_{}.h5'.format(seed,k,mode) in os.listdir('.'):
    model.load_weights('model_{}_{}_{}.h5'.format(seed,k,mode))

#for i in range(10):
#    if no_improve >= 20:
#        break
#    history = model.fit([enc_in,dec_in],dec_out,batch_size=batch_size,validation_split=.2,epochs=1)
#    model.save_weights('model_{}_{}_{}.h5'.format(seed,k,mode))
#    print('epoch {} over'.format(i))
#    K.clear_session()
#    val_loss = history.history['val_loss'][0]
#    if val_loss >= val_loss_old:
#        no_improve += 1
#    else:
#        no_improve = 0
#    val_loss_old = val_loss
#    f = open('loss_no_improvement_{}_{}_{}.txt'.format(seed,k,mode),'w')
#    print(str(no_improve)+' '+str(val_loss_old),file=f)
#    f.close()

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
        if mode == 'null':
            output_tokens = model.predict([input_seq,target_seq])
        curr_index = np.argmax(output_tokens[:,t,:])
        target_seq[0, t+1, curr_index] = 1.
        symbol = output_segs[curr_index]
        string.append(symbol)
        if symbol == '$':
            break
    return(string)


ind = 0
if 'decoded_{}_{}_{}.tsv'.format(seed,k,mode) in os.listdir('.'):
    ind = len([l for l in open('decoded_{}_{}_{}.tsv'.format(seed,k,mode),'r')])

for l,p,s,i,o in zip(lang_raw['test'][ind:],POS_raw['test'][ind:],gloss_raw['test'][ind:],input_raw['test'][ind:],output_raw['test'][ind:]):
    f = open('decoded_{}_{}_{}.tsv'.format(seed,k,mode),'a')
    o_ = decode_sequence(i,l,p,s,langs,output_segs,input_segs,POS_segs,L,X,Y,J,S,T_x,T_y,model,mode)
    d = editdistance.distance(o,o_)/max([len(o),len(o_)])
    print('\t'.join([l,''.join(i),''.join(o),''.join(o_),str(d)]),file=f)
    f.close()
