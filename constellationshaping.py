# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
import json
import math
import comm_operation as comm

# global variable
log_2 = math.log(2.0)

# the matrixs can convert one_hot symbol to binary code
sym2bin_map16 = []
for x in range(16):
    tmp_list = []
    for _ in range(4):
        tmp_list.append(x % 2)
        x //= 2
    tmp_list.reverse()
    sym2bin_map16.append(tmp_list)
sym2bin_map16 = np.array(sym2bin_map16, dtype=np.float32)

sym2bin_map64 = []
for x in range(64):
    tmp_list = []
    for _ in range(6):
        tmp_list.append(x % 2)
        x //= 2
    tmp_list.reverse()
    sym2bin_map64.append(tmp_list)
sym2bin_map64 = np.array(sym2bin_map64, dtype=np.float32)

sym2bin_map256 = []
for x in range(256):
    tmp_list = []
    for _ in range(8):
        tmp_list.append(x % 2)
        x //= 2
    tmp_list.reverse()
    sym2bin_map256.append(tmp_list)
sym2bin_map256 = np.array(sym2bin_map256, dtype=np.float32)
mapper_dict = {16: sym2bin_map16, 64: sym2bin_map64, 256: sym2bin_map256}


@tf.custom_gradient
def ste(x):
    M = x.shape[-1]
    y = tf.one_hot(tf.argmax(x, axis=-1), depth=M)

    def grad(dy):
        return dy*0

    return y, grad


# @tf.custom_gradient
# def sym2bin(x,matrix):
#     M = int(x.shape[-1])
#     y = tf.one_hot(tf.argmax(x, axis=-1), depth=M)
#     y = tf.matmul(y, matrix)
#
#     def grad(dy):
#         return (dy * 1,dy*1)
#
#     return y, grad


@tf.custom_gradient
def tanh_nonlinear(x):
    y = tf.tanh(x)

    def grad(dy):
        return dy

    return y, grad


def standard_gumbel(shape):
    t = keras.backend.random_uniform(shape=shape, minval=0.0, maxval=1.0)
    y = -keras.backend.log(-keras.backend.log(t))
    return y


class AwgnChannel(layers.Layer):

    def __init__(self, **kwargs):
        super(AwgnChannel, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return layers.Add()(inputs)


class NonlinearChannelKnown(layers.Layer):
    def __init__(self, **kwargs):
        super(NonlinearChannelKnown, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y = tf.tanh(inputs)
        return y


class NonlinearChannelUnkown(layers.Layer):
    def __init__(self, **kwargs):
        super(NonlinearChannelUnkown, self).__init__()

    def call(self, inputs, **kwargs):
        y = tanh_nonlinear(inputs)
        return y


class EntropyBit(layers.Layer):

    def __init__(self, **kwargs):
        super(EntropyBit, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        p1 = inputs[0]
        p2 = inputs[1]
        entropy = keras.losses.categorical_crossentropy(p1, p2) / log_2
        return entropy


class GumbelSampler(layers.Layer):
    def __init__(self, **kwargs):
        super(GumbelSampler, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y = ste(inputs)
        return y


class BinaryOutGumbelSampler(layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryOutGumbelSampler, self).__init__()

    def build(self, input_shape):
        M=int(input_shape[-1])
        self.mapper = tf.Variable(initial_value=mapper_dict[M],trainable=False,name='sym2bin_mapper')


    def call(self, inputs, **kwargs):
        y = ste(inputs)
        y = tf.matmul(y,self.mapper)
        return y


class AddLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AddLossLayer, self).__init__()

    def call(self, inputs, **kwargs):
        y_true = inputs[0]
        y_pred = inputs[1]
        prob_s = inputs[2]
        loss = keras.losses.binary_crossentropy(y_true, y_pred)++ 1 * tf.math.reduce_sum(
            tf.reduce_mean(tf.math.multiply(prob_s, tf.math.log(tf.math.maximum(prob_s,1e-7))), axis=0))
        return loss


class BaseConstellationShaping:
    def __init__(self, M):
        self.M = M
        self.bits_per_sym=int(math.log2(self.M))
    

class SymbolwiseShaping(BaseConstellationShaping):
    def __init__(self,M,snr,epochs,cce_weight=0.0,bce_weight=1.0,is_ebn0=False):
        super(SymbolwiseShaping,self).__init__(M=M)
        self.esn0_snr=snr
        if is_ebn0:
            self.esn0_snr=self.esn0_snr+10*math.log10(self.bits_per_sym)
        self.noise_sigma = math.sqrt(0.5/math.pow(10,self.esn0_snr/10))
        self.epochs=epochs
        self.batch_size=1500
        self.steps_per_epoch=1000
        self.prob_layers=5
        self.prob_nodes=120
        self.mapper_layers=5
        self.mapper_nodes=120
        self.learning_rate=1e-4
        self.decode_layers=5
        self.decode_nodes=120
        self.cce_weight=cce_weight
        self.bce_weight=bce_weight
        self.snr_inputs=keras.Input(shape=1,name="snr_inputs")
        self.gumbel_inputs=keras.Input(shape=self.M,name="gumbel_inputs")
        self.noise_inputs=keras.Input(shape=2,name="noise_inputs")

    def _prob(self):
        logits=layers.Dense(self.prob_nodes,activation='relu')(self.snr_inputs)
        for _ in range(self.prob_layers-1):
            logits=layers.Dense(self.prob_nodes,activation='relu')(logits)
        self.logits=layers.Dense(self.M,activation='linear',name="logits")(logits)
        self.prob=layers.Softmax(name="softmax_prob")(self.logits)
    
    def _symbols(self):
        self.sym_sample = layers.Softmax()(self.gumbel_inputs + self.logits)
        self.sym_dec = tf.argmax(self.sym_sample,axis=-1)
        self.sym_onehot = GumbelSampler()(self.sym_sample)
        self.binary_sym = BinaryOutGumbelSampler(name='binary_symbols')(self.sym_sample)
        self.sym_concat = layers.Concatenate(axis=-1)([self.sym_onehot,self.binary_sym])
    
    def _cons_mapper(self):
        in_phase=layers.Dense(self.mapper_nodes,activation='relu')(self.snr_inputs)
        quadra_phase = layers.Dense(self.mapper_nodes,activation='relu')(self.snr_inputs)
        for _ in range(self.mapper_layers-1):
            in_phase=layers.Dense(self.mapper_nodes,activation='relu')(in_phase)
            quadra_phase=layers.Dense(self.mapper_nodes,activation='relu')(quadra_phase)
        in_phase=layers.Dense(self.M+self.bits_per_sym,activation='linear')(in_phase)
        quadra_phase=layers.Dense(self.M+self.bits_per_sym,activation='linear')(quadra_phase)
        self.in_phase=tf.reduce_mean(in_phase,axis=0)
        self.quadra_phase=tf.reduce_mean(quadra_phase,axis=0)
        self.in_phase=tf.reshape(self.in_phase,shape=[-1,1])
        self.quadra_phase=tf.reshape(self.quadra_phase,shape=[-1,1])
        self.mapper_unnorm=layers.Concatenate(axis=-1)([self.in_phase,self.quadra_phase])
        base_sym_onehot = keras.utils.to_categorical(np.arange(self.M),num_classes=self.M)
        base_sym_bin = np.matmul(base_sym_onehot,mapper_dict[self.M])
        base_sym_concat = np.concatenate((base_sym_onehot,base_sym_bin),axis=-1)
        base_cons = tf.matmul(base_sym_concat,self.mapper_unnorm) # shape: (M,2)
        energy = tf.reshape(tf.reduce_sum(tf.math.square(base_cons),axis=-1),shape=[-1,1])
        energy = tf.reduce_mean(tf.matmul(self.prob,energy))
        self.mapper_norm = tf.divide(self.mapper_unnorm,tf.math.sqrt(energy))
        self.tx_norm = tf.matmul(self.sym_concat,self.mapper_norm) # shape:(batch_size,2)
    
    def _channel(self):
        self.rx = layers.Add(name="channel")([self.tx_norm,self.noise_inputs])
    
    def _decode_softmax(self):
        d_temp = layers.Dense(self.decode_nodes,activation='relu')(self.rx)
        for _ in range(self.decode_layers-1):
            d_temp=layers.Dense(self.decode_nodes,activation='relu')(d_temp)
        self.decode_out_softmax=layers.Dense(self.M,activation='softmax')(d_temp)
        self.decode_hard_sym = tf.argmax(self.decode_out_softmax,axis=-1)
        self.ser = keras.backend.mean(self.decode_hard_sym!=self.sym_dec)
        decode_onehot = tf.one_hot(self.decode_hard_sym,self.M)
        sym2bin = tf.convert_to_tensor(mapper_dict[self.M])
        hard_bits = tf.matmul(decode_onehot,sym2bin)
        self.ber_from_sym = keras.backend.mean(hard_bits!=self.binary_sym)        

    def _decode_sigmoid(self):
        d_temp = layers.Dense(self.decode_nodes,activation='relu')(self.rx)
        for _ in range(self.decode_layers-1):
            d_temp=layers.Dense(self.decode_nodes,activation='relu')(d_temp)
        self.decode_out_sigmoid = layers.Dense(self.bits_per_sym,activation='sigmoid')(d_temp)
        self.decode_hard_bits = tf.math.rint(self.decode_out_sigmoid)
        self.ber = keras.backend.mean(self.decode_hard_bits!=self.binary_sym)
        self.ser_from_bits = tf.reduce_mean(tf.math.reduce_prod(tf.cast(self.decode_hard_bits!=self.binary_sym,tf.float32),axis=-1))
    
    def create_model(self):
        self._prob()
        self._symbols()
        self._cons_mapper()
        self._channel()
        self._decode_softmax()
        self._decode_sigmoid()
        self.model_whole = keras.Model(inputs=[self.snr_inputs,self.gumbel_inputs,self.noise_inputs],outputs=[self.decode_out_softmax,self.decode_out_sigmoid])
        self.model_whole:keras.Model
        cce=keras.losses.categorical_crossentropy(y_true=self.sym_onehot,y_pred=self.decode_out_softmax)
        bce=keras.losses.binary_crossentropy(y_true=self.binary_sym,y_pred=self.decode_out_sigmoid)
        entropy_log = keras.losses.categorical_crossentropy(self.prob,self.prob)
        mutual_info_bit = tf.divide(cce-entropy_log,log_2)
        entropy_bits=tf.divide(entropy_log,log_2)
        mix_loss = self.cce_weight*cce+self.bce_weight*bce-entropy_log
        self.model_whole.add_loss(mix_loss)
        self.model_whole.add_metric(mutual_info_bit,name="mutual_info")
        self.model_whole.add_metric(entropy_bits,name="entropy")
        self.model_whole.add_metric(self.ser,name="ser")
        self.model_whole.add_metric(self.ber_from_sym,name="ber_from_sym")
        self.model_whole.add_metric(self.ber,name="ber")
        self.model_whole.add_metric(self.ser_from_bits,name="ser_from_bits")
        self.model_whole.compile(optimizer=keras.optimizers.Adam(self.learning_rate))
    
    def infinity_data_generator(self):
        snr_data = tf.ones(shape=[self.batch_size,1],dtype='float32')
        while True:
            gumbel_data = standard_gumbel(shape=[self.batch_size,self.M])
            noise = tf.random.normal(shape=[self.batch_size,2],stddev=self.noise_sigma)
            yield [snr_data,gumbel_data,noise]
    
    def train(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath="./model/test1/",monitor="mutual_info",verbose=1,save_best_only=True,mode='max'),
            keras.callbacks.EarlyStopping(monitor="mutual_info",patience=2,verbose=1,mode='max')
        ]
        self.model_whole.fit(self.infinity_data_generator(),epochs=self.epochs,steps_per_epoch=self.steps_per_epoch,callbacks=callbacks,verbose=2)
    
    def evaluate(self):
        pass
    
    def cons_prob_plot(self):
        pass
    


class BitwiseShaping(BaseConstellationShaping):
    def __init__(self, M, snr, EborEs):
        super(BitwiseShaping, self).__init__(M)

        self.prob_nodes = 16
        self.prob_layers = 5
        self.mapper_nodes = 64
        self.mapper_layers = 5
        self.decode_nodes = 64
        self.decode_layers = 5
        self.bit_per_sym = int(math.log2(self.M))
        self.train_batch_size = 2000
        self.epochs = 50
        self.steps_per_epoch = 1000
        self.learning_rate = 1e-3

        self.EborEs = EborEs
        self.snr_input = keras.Input(batch_shape=[self.train_batch_size, 1], name='snr_input')
        self.gumbel_input = keras.Input(batch_shape=[self.train_batch_size, self.M], name='gumbel_inputs')
        self.noise = keras.Input(batch_shape=[self.train_batch_size, 2], name='noise')

        if EborEs:
            self.snr = snr + 10 * math.log10(self.bit_per_sym)
        else:
            self.snr = snr
        self.noise_sigma = math.sqrt(0.5 / math.pow(10, self.snr / 10))

    def _prob_shaping(self):
        logits = layers.Dense(self.prob_nodes, activation='relu')(self.snr_input)
        for _ in range(self.prob_layers - 1):
            logits = layers.Dense(self.prob_nodes, activation='relu')(logits)
        self.logits = layers.Dense(self.M, activation='linear', name='logits')(logits)
        self.prob = layers.Softmax(name='prob_s')(self.logits)

    def _symbols(self):
        self.sym = layers.Softmax()(self.gumbel_input + self.logits)
        self.binary_sym = BinaryOutGumbelSampler(name='binary_symbols')(self.sym)

    def _cons_mapper(self):
        e_temp = layers.Dense(self.mapper_nodes, activation='relu')(self.binary_sym)
        for _ in range(self.mapper_layers - 1):
            e_temp = layers.Dense(self.mapper_nodes, activation='relu')(e_temp)
        self.tx = layers.Dense(2, activation='linear', name='tx_nonnormalized')(e_temp)
        prob = tf.reshape(self.prob[0, :], shape=[-1, 1])
        self.sym=tf.one_hot(tf.argmax(self.sym,axis=-1),depth=self.M)
        self.sym_prob = tf.matmul(self.sym, prob)
        tmp1=tf.reshape(tf.reduce_sum(tf.square(self.tx), axis=-1),shape=[-1,1])
        tx_energy = tf.reduce_sum(tf.multiply(tmp1, self.sym_prob))
        self.tx_norm = self.tx / tf.sqrt(tx_energy)

    def _channel(self):
        self.rx = layers.Add(name='awgn_channel')([self.tx_norm, self.noise])

    def _decoder(self):
        d_temp = layers.Dense(self.decode_nodes, activation='relu')(self.rx)
        for _ in range(self.decode_layers - 1):
            d_temp = layers.Dense(self.decode_nodes, activation='relu')(d_temp)

        self.decode_out = layers.Dense(self.bit_per_sym, activation='sigmoid', name='decode_out')(d_temp)

    def create_model(self):
        self._prob_shaping()
        self._symbols()
        self._cons_mapper()
        self._channel()
        self._decoder()
        # bce = keras.losses.BinaryCrossentropy()(self.binary_sym, self.decode_out)
        custom_loss = AddLossLayer()([self.binary_sym, self.decode_out,self.prob])
        self.hard_dicision_bit = tf.math.rint(self.decode_out)
        ber = keras.backend.mean(self.hard_dicision_bit != self.binary_sym)
        self.model = keras.Model(inputs=[self.snr_input, self.gumbel_input, self.noise], outputs=self.decode_out)
        self.model: keras.Model
        self.model.add_loss(custom_loss)
        self.model.add_metric(ber, name='ber')
        self.model.compile(optimizer=keras.optimizers.Adam(self.learning_rate))
        self.logits_model = keras.Model(inputs=self.snr_input,outputs=self.logits)
        self.tx_model = keras.Model(inputs=[self.snr_input,self.gumbel_input],outputs=self.tx_norm)

    def train_data_generator(self):
        snr_data = self.snr * np.ones([self.train_batch_size, 1], dtype=np.float32)
        while True:
            gumbel_data = standard_gumbel(shape=[self.train_batch_size, self.M])
            noise = tf.random.normal(mean=0, stddev=self.noise_sigma, shape=[self.train_batch_size, 2])
            yield [snr_data, gumbel_data, noise]

    def train(self):
        # snr_data = self.snr * np.ones([self.train_batch_size * self.steps_per_epoch, 1], dtype=np.float32)
        # gumbel_data = standard_gumbel(shape=[self.train_batch_size * self.steps_per_epoch, self.M])
        # noise = tf.random.normal(mean=0, stddev=self.noise_sigma,
        #                          shape=[self.train_batch_size * self.steps_per_epoch, 2])
        # print(self.binary_sym)
        cbs = [
            keras.callbacks.EarlyStopping(monitor='loss',patience=2,verbose=1,mode='min')
        ]
        self.model.fit(self.train_data_generator(),batch_size=self.train_batch_size, epochs=self.epochs,steps_per_epoch=self.steps_per_epoch)
        # self.model.fit(x={'snr_input': snr_data, 'gumbel_inputs': gumbel_data, 'noise': noise},
        #                epochs=self.epochs, batch_size=self.train_batch_size)

    def cons_point_generate(self):
        snr_d = self.snr*np.ones([self.M,1],dtype=np.float32)
        logits_out = self.logits_model(snr_d).numpy()
        gumbel_d = keras.utils.to_categorical(np.arange(self.M),self.M)
        self.sym_model = keras.Model(inputs=[self.snr_input,self.gumbel_input],outputs=self.sym)
        sym = self.sym_model([snr_d,gumbel_d])
        tx_norm = self.tx_model([snr_d,gumbel_d]).numpy()
        prob_s = tf.math.softmax(logits_out)
        prob_s = np.array(prob_s[0]).reshape([-1])
        s = prob_s*100
        print(logits_out)
        print(sym)
        plt.scatter(tx_norm[:,0],tx_norm[:,1],s=s)
        plt.show()
        plt.plot(prob_s)
        plt.show()


if __name__ == '__main__':
    comm.set_device(gpu_id=1,gpu_mem=4096)
    comm.redirect2log('log.log')
    print("tf version:",tf.__version__)
    symwise_shaper = SymbolwiseShaping(M=16,snr=6,epochs=10,cce_weight=0.5,bce_weight=1.0,is_ebn0=False)
    symwise_shaper.create_model()
    symwise_shaper.train()
    # bitwise_shaper = BitwiseShaping(M=64, snr=14, EborEs=True)
    # bitwise_shaper.create_model()
    # bitwise_shaper.model.summary()
    # # keras.utils.plot_model(bitwise_shaper.model,show_shapes=True)
    # # bitwise_shaper.model.evaluate(bitwise_shaper.train_data_generator(), steps=10)
    # bitwise_shaper.train()
    # bitwise_shaper.cons_point_generate()

    # debug
    # 前向可以跑通
    # snr_data = 10 * np.ones([100, 1], dtype=np.float32)
    # gumbel_data = standard_gumbel(shape=[100, bitwise_shaper.M])
    # noise = tf.random.normal(mean=0, stddev=bitwise_shaper.noise_sigma,
    #                          shape=[100, 2])
    # d_out = bitwise_shaper.model([snr_data,gumbel_data,noise])
    # print(d_out)
