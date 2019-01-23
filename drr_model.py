#! -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras import initializers
from tensorflow.python.keras.activations import tanh, softmax
from tensorflow.python.keras.layers import Add, Conv1D, Lambda, Dropout, Dense, GRU, LSTM, InputSpec, Bidirectional, TimeDistributed, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.layers import Layer, Input, concatenate, GlobalAveragePooling1D, Embedding,  RepeatVector, Reshape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.initializers import Ones, Zeros
import numpy as np

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn

def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=2, dropout=0.1):
        self.emb_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
    def __call__(self, x, return_att=False, mask=None, active_layers=999):
        x = self.emb_dropout(x)
        if return_att: atts = []
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x

class DrrModel:
    def __init__(self, seq_len, d_feature, d_model=64, d_inner_hid=128, n_head=1, d_k=64, d_v=64, layers=2, dropout=0.1):
        self.seq_len = seq_len
        self.d_feature = d_feature
        self.d_model = d_model
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout)
    #drr_base or drr_personalized_v2
    def build_model(self, pos_mode=0, use_mask=False, active_layers=999):
        v_input = Input(shape=(self.seq_len, self.d_feature), name='v_input')
        d0 = TimeDistributed(Dense(self.d_model))(v_input)
        pos_input = Input(shape=(self.seq_len,), dtype='int32', name='pos_input')
        if pos_mode == 0:  # use fixed pos embedding
            pos_embedding = Embedding(self.seq_len, self.d_model, trainable=False,\
                weights=[GetPosEncodingMatrix(self.seq_len, self.d_model)])
            p0 = pos_embedding(pos_input)
        elif pos_mode == 1: # use trainable pos embedding
            pos_embedding = Embedding(self.seq_len, self.d_model)
            p0 = pos_embedding(pos_input)
        else:  # no pos embedding
            p0 = None
        if p0 != None:
            combine_input = Add()([d0, p0])
        else:
            combine_input = d0 # no pos
        sub_mask = None
        if use_mask:
            sub_mask = Lambda(GetSubMask)(pos_input)
        enc_output = self.encoder(combine_input, mask=sub_mask, active_layers=active_layers)
        # score
        time_score_dense1 = TimeDistributed(Dense(self.d_model, activation='tanh'))(enc_output)
        time_score_dense2 = TimeDistributed(Dense(1))(time_score_dense1)
        flat = Flatten()(time_score_dense2)
        score_output = Activation(activation='softmax')(flat)
        self.model = Model([pos_input, v_input], score_output)
        return self.model
    #drr_personalized_v1
    def build_model_ex(self, pos_mode=0, use_mask=False, active_layers=999):
        #define embedding layer
        uid_embedding = Embedding(750000, 16, name='uid_embedding') # for uid
        itemid_embedding = Embedding(7500000, 32, name='itemid_embedding') # for icf1
        f1_embedding = Embedding(8, 2, name='f1_embedding') # for ucf1
        f2_embedding = Embedding(4, 2, name='f2_embedding') # for ucf2 & icf3 
        f3_embedding = Embedding(8, 2, name='f3_embedding') # for ucf3 & icf4
        f4_embedding = Embedding(4, 2, name='f4_embedding') # for icf5
        f5_embedding = Embedding(256, 4, name='f5_embedding') # icf2
        #define user input
        uid_input = Input(shape=(self.seq_len,), dtype='int32', name='uid_input')
        ucf1_input = Input(shape=(self.seq_len,), dtype='int32', name='ucf1_input')
        ucf2_input = Input(shape=(self.seq_len,), dtype='int32', name='ucf2_input')
        ucf3_input = Input(shape=(self.seq_len,), dtype='int32', name='ucf3_input')
        #define item input
        icf1_input = Input(shape=(self.seq_len,), dtype='int32', name='icf1_input')
        icf2_input = Input(shape=(self.seq_len,), dtype='int32', name='icf2_input')
        icf3_input = Input(shape=(self.seq_len,), dtype='int32', name='icf3_input')
        icf4_input = Input(shape=(self.seq_len,), dtype='int32', name='icf4_input')
        icf5_input = Input(shape=(self.seq_len,), dtype='int32', name='icf5_input')
        #define dense input
        v_input = Input(shape=(self.seq_len, self.d_feature), name='v_input')
        #define user embedding
        u0 = uid_embedding(uid_input)
        u1 = f1_embedding(ucf1_input)
        u2 = f2_embedding(ucf2_input)
        u3 = f3_embedding(ucf3_input)
        #define item embedding
        i1 = itemid_embedding(icf1_input)
        i2 = f5_embedding(icf2_input)
        i3 = f2_embedding(icf3_input)
        i4 = f3_embedding(icf4_input)
        i5 = f4_embedding(icf5_input)
        #define page embedding: 16+2+2+2+32+4+2+2+2=64
        page_embedding = concatenate([v_input, u0, u1, u2, u3, i1, i2, i3, i4, i5], axis=-1, name='page_embedding')
        d0 = TimeDistributed(Dense(self.d_model))(page_embedding)
        pos_input = Input(shape=(self.seq_len,), dtype='int32', name='pos_input')
        if pos_mode == 0:  # use fix pos embedding
            pos_embedding = Embedding(self.seq_len, self.d_model, trainable=False,\
                weights=[GetPosEncodingMatrix(self.seq_len, self.d_model)])
            p0 = pos_embedding(pos_input)
        elif pos_mode == 1: # use trainable ebmedding
            pos_embedding = Embedding(self.seq_len, self.d_model)
            p0 = pos_embedding(pos_input)
        else:  # not use pos embedding
            p0 = None
        if p0 != None:
            combine_input = Add()([d0, p0])
        else:
            combine_input = d0 # no pos
        sub_mask = None
        if use_mask:
            sub_mask = Lambda(GetSubMask)(pos_input)
        enc_output = self.encoder(combine_input, mask=sub_mask, active_layers=active_layers)
        # score
        time_score_dense1 = TimeDistributed(Dense(self.d_model, activation='tanh'))(enc_output)
        time_score_dense2 = TimeDistributed(Dense(1))(time_score_dense1)
        flat = Flatten()(time_score_dense2)
        score_output = Activation(activation='softmax')(flat)
        base_input = [pos_input, uid_input, ucf1_input, ucf2_input, ucf3_input, icf1_input, icf2_input, icf3_input, icf4_input, icf5_input, v_input]
        self.model = Model(base_input, score_output)
        return self.model
