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


assert(len(sys.argv) == 4)
seed = int(sys.argv[1])
k = int(sys.argv[2])
mode = sys.argv[3]



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

if mode == 'lang':
    rep_latent_inputs_ = RepeatVector(T_x)(embed_lang_in_)

if mode == 'langPOS':
    rep_latent_inputs_ = RepeatVector(T_x)(Concatenate()([embed_lang_in_,embed_POS_in_]))

if mode == 'langPOSsem':
    rep_latent_inputs_ = RepeatVector(T_x)(Concatenate()([embed_lang_in_,embed_POS_in_,embed_sem_in_]))

if mode == 'langPOSsemetym':
    rep_latent_inputs_ = RepeatVector(T_x)(Concatenate()([embed_lang_in_,embed_POS_in_,embed_sem_in_,embed_etym_in_]))

embed_inputs_ = Concatenate()([enc_in_,rep_latent_inputs_])
embedding = Dense(embed_dim)(embed_inputs_)

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

if mode == 'lang':
    alignment_model = Model([embed_lang_in_,enc_in_,dec_in_],alignment_probs_)
    alignment_output = alignment_model([embed_lang(lang_id_),enc_in_,dec_in_])
    alignment = Model([lang_id_,enc_in_,dec_in_],alignment_output)

if mode == 'langPOS':
    alignment_model = Model([embed_lang_in_,embed_POS_in_,enc_in_,dec_in_],alignment_probs_)
    alignment_output = alignment_model([embed_lang(lang_id_),embed_POS(POS_in_),enc_in_,dec_in_])
    alignment = Model([lang_id_,POS_in_,enc_in_,dec_in_],alignment_output)

if mode == 'langPOSsem':
    alignment_model = Model([embed_lang_in_,embed_POS_in_,embed_sem_in_,enc_in_,dec_in_],alignment_probs_)
    alignment_output = alignment_model([embed_lang(lang_id_),embed_POS(POS_in_),embed_sem(sem_in_),enc_in_,dec_in_])
    alignment = Model([lang_id_,POS_in_,sem_in_,enc_in_,dec_in_],alignment_output)

if mode == 'langPOSsemetym':
    alignment_model = Model([embed_lang_in_,embed_POS_in_,embed_sem_in_,embed_etym_in_,enc_in_,dec_in_],alignment_probs_)
    alignment_output = alignment_model([embed_lang(lang_id_),embed_POS(POS_in_),embed_sem(sem_in_),embed_etym([enc_in_,sem_in_]),enc_in_,dec_in_])
    alignment = Model([lang_id_,POS_in_,sem_in_,enc_in_,dec_in_],alignment_output)

alphas = tf.expand_dims(alignment_probs_,-1)*emission_probs
dec_out_ = tf.reduce_sum(alphas,-3)
dec_out_ = Lambda(lambda x:x)(dec_out_)

if mode == 'lang':
    decoder = Model([embed_lang_in_,enc_in_,dec_in_],dec_out_)
    output = decoder([embed_lang(lang_id_),enc_in_,dec_in_])
    model = Model(inputs=[lang_id_,enc_in_,dec_in_], outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    if 'model_{}_{}_{}.h5'.format(seed,k,mode) in os.listdir('.'):
        model.load_weights('model_{}_{}_{}.h5'.format(seed,k,mode))

if mode == 'langPOS':
    decoder = Model([embed_lang_in_,embed_POS_in_,enc_in_,dec_in_],dec_out_)
    output = decoder([embed_lang(lang_id_),embed_POS(POS_in_),enc_in_,dec_in_])
    model = Model(inputs=[lang_id_,POS_in_,enc_in_,dec_in_], outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    if 'model_{}_{}_{}.h5'.format(seed,k,mode) in os.listdir('.'):
        model.load_weights('model_{}_{}_{}.h5'.format(seed,k,mode))

if mode == 'langPOSsem':
    decoder = Model([embed_lang_in_,embed_POS_in_,embed_sem_in_,enc_in_,dec_in_],dec_out_)
    output = decoder([embed_lang(lang_id_),embed_POS(POS_in_),embed_sem(sem_in_),enc_in_,dec_in_])
    model = Model(inputs=[lang_id_,POS_in_,sem_in_,enc_in_,dec_in_], outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    if 'model_{}_{}_{}.h5'.format(seed,k,mode) in os.listdir('.'):
        model.load_weights('model_{}_{}_{}.h5'.format(seed,k,mode))

if mode == 'langPOSsemetym':
    decoder = Model([embed_lang_in_,embed_POS_in_,embed_sem_in_,embed_etym_in_,enc_in_,dec_in_],dec_out_)
    output = decoder([embed_lang(lang_id_),embed_POS(POS_in_),embed_sem(sem_in_),embed_etym([enc_in_,sem_in_]),enc_in_,dec_in_])
    model = Model(inputs=[lang_id_,POS_in_,sem_in_,enc_in_,dec_in_], outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    if 'model_{}_{}_{}.h5'.format(seed,k,mode) in os.listdir('.'):
        model.load_weights('model_{}_{}_{}.h5'.format(seed,k,mode))


embeddings = embed_lang(np.eye(L))

f = open('tree_stuff/language_embeddings_{}_{}_{}.tsv'.format(seed,k,mode),'w')
for i,l in enumerate(langs):
    print('\t'.join([l]+[str(float(s)) for s in embeddings[i,]]),file=f)


f.close()

