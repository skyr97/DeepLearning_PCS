# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
import os
import shutil
import json
import math
import comm_operation as comm

# global variable
LOG_2 = math.log(2.0)

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
        entropy = keras.losses.categorical_crossentropy(p1, p2) / LOG_2
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
        M = int(input_shape[-1])
        self.mapper = tf.Variable(
            initial_value=mapper_dict[M], trainable=False, name='sym2bin_mapper')

    def call(self, inputs, **kwargs):
        y = ste(inputs)
        y = tf.matmul(y, self.mapper)
        return y


class AddLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AddLossLayer, self).__init__()

    def call(self, inputs, **kwargs):
        y_true = inputs[0]
        y_pred = inputs[1]
        prob_s = inputs[2]
        loss = keras.losses.binary_crossentropy(y_true, y_pred)++ 1 * tf.math.reduce_sum(
            tf.reduce_mean(tf.math.multiply(prob_s, tf.math.log(tf.math.maximum(prob_s, 1e-7))), axis=0))
        return loss


class BaseConstellationShaping:
    def __init__(self, M):
        self.M = M
        self.bits_per_sym = int(math.log2(self.M))


class SymbolwiseShaping(BaseConstellationShaping):
    def __init__(self, M, snr, epochs, cce_weight=0.0, bce_weight=1.0, is_ebn0=False):
        super(SymbolwiseShaping, self).__init__(M=M)
        self.esn0_snr = snr
        if is_ebn0:
            self.esn0_snr = self.esn0_snr+10*math.log10(self.bits_per_sym)
        self.noise_sigma = math.sqrt(0.5/math.pow(10, self.esn0_snr/10))
        self.epochs = epochs
        self.batch_size = 1500
        self.steps_per_epoch = 1000
        self.prob_layers = 5
        self.prob_nodes = 120
        self.mapper_layers = 5
        self.mapper_nodes = 120
        self.learning_rate = 1e-4
        self.decode_layers = 5
        self.decode_nodes = 120
        self.cce_weight = cce_weight
        self.bce_weight = bce_weight
        self.snr_inputs = keras.Input(shape=1, name="snr_inputs")
        self.gumbel_inputs = keras.Input(shape=self.M, name="gumbel_inputs")
        self.noise_inputs = keras.Input(shape=2, name="noise_inputs")

    def _prob(self):
        logits = layers.Dense(
            self.prob_nodes, activation='relu')(self.snr_inputs)
        for _ in range(self.prob_layers-1):
            logits = layers.Dense(self.prob_nodes, activation='relu')(logits)
        self.logits = layers.Dense(
            self.M, activation='linear', name="logits")(logits)
        self.prob = layers.Softmax(name="softmax_prob")(self.logits)
        self.prob_model = keras.Model(
            inputs=self.snr_inputs, outputs=self.prob)

    def _symbols(self):
        self.sym_sample = layers.Softmax()(self.gumbel_inputs + self.logits)
        self.sym_dec = tf.argmax(self.sym_sample, axis=-1)
        self.sym_onehot = GumbelSampler()(self.sym_sample)
        self.binary_sym = BinaryOutGumbelSampler(
            name='binary_symbols')(self.sym_sample)
        self.sym_concat = layers.Concatenate(
            axis=-1)([self.sym_onehot, self.binary_sym])

    def _cons_mapper(self):
        self.map_mode = "concat"
        in_phase = layers.Dense(
            self.mapper_nodes, activation='relu')(self.snr_inputs)
        quadra_phase = layers.Dense(
            self.mapper_nodes, activation='relu')(self.snr_inputs)
        for _ in range(self.mapper_layers-1):
            in_phase = layers.Dense(
                self.mapper_nodes, activation='relu')(in_phase)
            quadra_phase = layers.Dense(
                self.mapper_nodes, activation='relu')(quadra_phase)
        in_phase = layers.Dense(self.M+self.bits_per_sym,
                                activation='linear')(in_phase)
        quadra_phase = layers.Dense(
            self.M+self.bits_per_sym, activation='linear')(quadra_phase)
        self.in_phase = tf.reduce_mean(in_phase, axis=0)
        self.quadra_phase = tf.reduce_mean(quadra_phase, axis=0)
        self.in_phase = tf.reshape(self.in_phase, shape=[-1, 1])
        self.quadra_phase = tf.reshape(self.quadra_phase, shape=[-1, 1])
        self.mapper_unnorm = layers.Concatenate(
            axis=-1)([self.in_phase, self.quadra_phase])
        base_sym_onehot = keras.utils.to_categorical(
            np.arange(self.M), num_classes=self.M)
        base_sym_bin = np.matmul(base_sym_onehot, mapper_dict[self.M])
        base_sym_concat = np.concatenate(
            (base_sym_onehot, base_sym_bin), axis=-1)
        base_cons = tf.matmul(
            base_sym_concat, self.mapper_unnorm)  # shape: (M,2)
        energy = tf.reshape(tf.reduce_sum(
            tf.math.square(base_cons), axis=-1), shape=[-1, 1])
        energy = tf.reduce_mean(tf.matmul(self.prob, energy))
        self.mapper_norm = tf.divide(self.mapper_unnorm, tf.math.sqrt(energy))
        # shape:(batch_size,2)
        self.tx_norm = tf.matmul(self.sym_concat, self.mapper_norm)

    def _cons_mapper_onehot(self):
        self.map_mode = "onehot"
        in_phase = layers.Dense(
            self.mapper_nodes, activation='relu')(self.snr_inputs)
        quadra_phase = layers.Dense(
            self.mapper_nodes, activation='relu')(self.snr_inputs)
        for _ in range(self.mapper_layers-1):
            in_phase = layers.Dense(
                self.mapper_nodes, activation='relu')(in_phase)
            quadra_phase = layers.Dense(
                self.mapper_nodes, activation='relu')(quadra_phase)
        in_phase = layers.Dense(self.M//4, activation='linear')(in_phase)
        quadra_phase = layers.Dense(
            self.M//4, activation='linear')(quadra_phase)

        self.in_phase = tf.reduce_mean(in_phase, axis=0)
        self.quadra_phase = tf.reduce_mean(quadra_phase, axis=0)
        self.in_phase = tf.reshape(self.in_phase, shape=[-1, 1])
        self.quadra_phase = tf.reshape(self.quadra_phase, shape=[-1, 1])

        quadrarant_1st = layers.Concatenate(
            axis=-1)([self.in_phase, self.quadra_phase])
        quadrarant_2nd = layers.Concatenate(
            axis=-1)([-self.in_phase, self.quadra_phase])
        quadrarant_3th = layers.Concatenate(
            axis=-1)([-self.in_phase, -self.quadra_phase])
        quadrarant_4th = layers.Concatenate(
            axis=-1)([self.in_phase, -self.quadra_phase])
        # self.mapper_unnorm = layers.Concatenate(axis=-1)([self.in_phase, self.quadra_phase])
        self.mapper_unnorm = layers.Concatenate(axis=0)(
            [quadrarant_1st, quadrarant_2nd, quadrarant_3th, quadrarant_4th])
        base_sym_onehot = keras.utils.to_categorical(
            np.arange(self.M), num_classes=self.M)
        base_cons = tf.matmul(
            base_sym_onehot, self.mapper_unnorm)  # shape: (M,2)
        energy = tf.reshape(tf.reduce_sum(
            tf.math.square(base_cons), axis=-1), shape=[-1, 1])
        energy = tf.reduce_mean(tf.matmul(self.prob, energy))
        self.mapper_norm = tf.divide(self.mapper_unnorm, tf.math.sqrt(energy))
        # shape:(batch_size,2)
        self.tx_norm = tf.matmul(self.sym_onehot, self.mapper_norm)

    def _channel(self):
        self.rx = layers.Add(name="channel")([self.tx_norm, self.noise_inputs])

    def _decode_softmax(self):
        d_temp = layers.Dense(self.decode_nodes, activation='relu')(self.rx)
        for _ in range(self.decode_layers-1):
            d_temp = layers.Dense(self.decode_nodes, activation='relu')(d_temp)
        self.decode_out_softmax = layers.Dense(
            self.M, activation='softmax')(d_temp)
        self.decode_hard_sym = tf.argmax(self.decode_out_softmax, axis=-1)
        self.ser = keras.backend.mean(self.decode_hard_sym != self.sym_dec)
        decode_onehot = tf.one_hot(self.decode_hard_sym, self.M)
        sym2bin = tf.convert_to_tensor(mapper_dict[self.M])
        hard_bits = tf.matmul(decode_onehot, sym2bin)
        self.ber_from_sym = keras.backend.mean(hard_bits != self.binary_sym)

    def _decode_sigmoid(self):
        d_temp = layers.Dense(self.decode_nodes, activation='relu')(self.rx)
        for _ in range(self.decode_layers-1):
            d_temp = layers.Dense(self.decode_nodes, activation='relu')(d_temp)
        self.decode_out_sigmoid = layers.Dense(
            self.bits_per_sym, activation='sigmoid')(d_temp)
        self.decode_hard_bits = tf.math.rint(self.decode_out_sigmoid)
        self.ber = keras.backend.mean(self.decode_hard_bits != self.binary_sym)
        self.ser_from_bits = 1-tf.reduce_mean(tf.math.reduce_prod(
            tf.cast(self.decode_hard_bits != self.binary_sym, tf.float32), axis=-1))

    def create_model(self):
        self._prob()
        self._symbols()
        # self._cons_mapper()
        self._cons_mapper_onehot()
        self._channel()
        self._decode_softmax()
        self._decode_sigmoid()
        self.model_whole = keras.Model(inputs=[self.snr_inputs, self.gumbel_inputs, self.noise_inputs], outputs=[
                                       self.decode_out_softmax, self.decode_out_sigmoid])
        self.model_whole: keras.Model
        cce = keras.losses.categorical_crossentropy(
            y_true=self.sym_onehot, y_pred=self.decode_out_softmax)
        bce = keras.losses.binary_crossentropy(
            y_true=self.binary_sym, y_pred=self.decode_out_sigmoid)
        entropy_log = keras.losses.categorical_crossentropy(
            self.prob, self.prob)
        mutual_info_bit = -tf.divide(cce-entropy_log, LOG_2)
        entropy_bits = tf.divide(entropy_log, LOG_2)
        mix_loss = self.cce_weight*cce+self.bce_weight*bce-entropy_log
        self.model_whole.add_loss(mix_loss)
        self.model_whole.add_metric(mutual_info_bit, name="mutual_info")
        self.model_whole.add_metric(entropy_bits, name="entropy")
        self.model_whole.add_metric(self.ser, name="ser")
        self.model_whole.add_metric(self.ber_from_sym, name="ber_from_sym")
        self.model_whole.add_metric(self.ber, name="ber")
        self.model_whole.add_metric(self.ser_from_bits, name="ser_from_bits")
        self.model_whole.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate))

    def infinity_data_generator(self):
        snr_data = tf.ones(shape=[self.batch_size, 1], dtype='float32')
        while True:
            gumbel_data = standard_gumbel(shape=[self.batch_size, self.M])
            noise = tf.random.normal(
                shape=[self.batch_size, 2], stddev=self.noise_sigma)
            yield [snr_data, gumbel_data, noise]

    def train(self):
        model_path = "./model/modOrder{M}/snr{snr:.2f}".format(
            M=self.M, snr=self.esn0_snr)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=model_path, monitor="mutual_info", verbose=1, save_best_only=True, mode='max'),
            keras.callbacks.EarlyStopping(
                monitor="mutual_info", patience=3, verbose=1, mode='max')
        ]
        self.model_whole.fit(self.infinity_data_generator(), epochs=self.epochs,
                             steps_per_epoch=self.steps_per_epoch, callbacks=callbacks, verbose=2)

    def evaluate(self, matfilename):
        keys = ["loss", "mutual_info", "entropy", "ser",
                "ber_from_sym", "ber", "ser_from_bits"]
        vali_dict = dict()
        [vali_dict[keys[0]], vali_dict[keys[1]], vali_dict[keys[2]], vali_dict[keys[3]], vali_dict[keys[4]], vali_dict[keys[5]],
            vali_dict[keys[6]]] = self.model_whole.evaluate(self.infinity_data_generator(), verbose=2, steps=1000)
        # [loss,mutual_info,entropy,ser,ber_from_sym,ber,ser_from_bits]=self.model_whole.evaluate(self.infinity_data_generator(),verbose=2,steps=1000)
        vali_dict["snr"] = self.esn0_snr
        if os.path.exists(matfilename):
            mat_dict = loadmat(matfilename)
        else:
            mat_dict = dict()
        for k in vali_dict:
            if k not in mat_dict:
                mat_dict[k] = vali_dict[k]
            else:
                mat_dict[k] = np.append(mat_dict[k], vali_dict[k])
        savemat(matfilename, mat_dict)

    def cons_prob_plot(self):
        if self.map_mode == "concat":
            self.sym_inputs = keras.Input(
                shape=self.M+self.bits_per_sym, name="symbol_input")
        else:
            self.sym_inputs = keras.Input(shape=self.M, name="symbol_input")
        tx_norm = tf.matmul(self.sym_inputs, self.mapper_norm)
        self.tx_norm_model = keras.Model(
            inputs=[self.snr_inputs, self.sym_inputs], outputs=tx_norm)
        base_sym = np.arange(self.M)
        base_sym_onehot = keras.utils.to_categorical(base_sym, self.M)
        base_sym_bin = np.matmul(base_sym_onehot, mapper_dict[self.M])
        base_sym_concat = np.concatenate((base_sym_onehot, base_sym_bin), -1)
        snr_data = self.esn0_snr*np.ones(shape=[self.M, 1], dtype='float32')
        if self.map_mode == "concat":
            base_tx = self.tx_norm_model([snr_data, base_sym_concat]).numpy()
        else:
            base_tx = self.tx_norm_model([snr_data, base_sym_onehot]).numpy()
        prob_s = self.prob_model(snr_data).numpy()
        prob_s = prob_s[0]

        # print(base_tx)
        image_path = "./images/modOrder{M}".format(M=self.M)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        markers = []
        base_sym_bin = base_sym_bin.astype(np.int64)
        for bin in base_sym_bin:
            s = ""
            for b in bin:
                s += str(b)
            s = "$"+s+"$"
            markers.append(s)

        plt.scatter(base_tx[:, 0], base_tx[:, 1], s=prob_s*500)

        for i in range(len(base_tx)):
            plt.scatter(base_tx[i, 0], base_tx[i, 1],
                        s=prob_s[i]*7000, marker=markers[i], c='r')
        plt.savefig(os.path.join(
            image_path, "snr{snr:.2f}_order{M}.png".format(snr=self.esn0_snr, M=self.M)))
        plt.close()
        savemat(os.path.join(image_path, "snr{snr:.2f}_order{M}.mat".format(
            snr=self.esn0_snr, M=self.M)), {"prob": prob_s, "cons": base_tx})


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
        self.snr_input = keras.Input(
            batch_shape=[self.train_batch_size, 1], name='snr_input')
        self.gumbel_input = keras.Input(
            batch_shape=[self.train_batch_size, self.M], name='gumbel_inputs')
        self.noise = keras.Input(
            batch_shape=[self.train_batch_size, 2], name='noise')

        if EborEs:
            self.snr = snr + 10 * math.log10(self.bit_per_sym)
        else:
            self.snr = snr
        self.noise_sigma = math.sqrt(0.5 / math.pow(10, self.snr / 10))

    def _prob_shaping(self):
        logits = layers.Dense(
            self.prob_nodes, activation='relu')(self.snr_input)
        for _ in range(self.prob_layers - 1):
            logits = layers.Dense(self.prob_nodes, activation='relu')(logits)
        self.logits = layers.Dense(
            self.M, activation='linear', name='logits')(logits)
        self.prob = layers.Softmax(name='prob_s')(self.logits)

    def _symbols(self):
        self.sym = layers.Softmax()(self.gumbel_input + self.logits)
        self.binary_sym = BinaryOutGumbelSampler(
            name='binary_symbols')(self.sym)

    def _cons_mapper(self):
        e_temp = layers.Dense(
            self.mapper_nodes, activation='relu')(self.binary_sym)
        for _ in range(self.mapper_layers - 1):
            e_temp = layers.Dense(self.mapper_nodes, activation='relu')(e_temp)
        self.tx = layers.Dense(2, activation='linear',
                               name='tx_nonnormalized')(e_temp)
        prob = tf.reshape(self.prob[0, :], shape=[-1, 1])
        self.sym = tf.one_hot(tf.argmax(self.sym, axis=-1), depth=self.M)
        self.sym_prob = tf.matmul(self.sym, prob)
        tmp1 = tf.reshape(tf.reduce_sum(
            tf.square(self.tx), axis=-1), shape=[-1, 1])
        tx_energy = tf.reduce_sum(tf.multiply(tmp1, self.sym_prob))
        self.tx_norm = self.tx / tf.sqrt(tx_energy)

    def _channel(self):
        self.rx = layers.Add(name='awgn_channel')([self.tx_norm, self.noise])

    def _decoder(self):
        d_temp = layers.Dense(self.decode_nodes, activation='relu')(self.rx)
        for _ in range(self.decode_layers - 1):
            d_temp = layers.Dense(self.decode_nodes, activation='relu')(d_temp)

        self.decode_out = layers.Dense(
            self.bit_per_sym, activation='sigmoid', name='decode_out')(d_temp)

    def create_model(self):
        self._prob_shaping()
        self._symbols()
        self._cons_mapper()
        self._channel()
        self._decoder()
        # bce = keras.losses.BinaryCrossentropy()(self.binary_sym, self.decode_out)
        custom_loss = AddLossLayer()(
            [self.binary_sym, self.decode_out, self.prob])
        self.hard_dicision_bit = tf.math.rint(self.decode_out)
        ber = keras.backend.mean(self.hard_dicision_bit != self.binary_sym)
        self.model = keras.Model(
            inputs=[self.snr_input, self.gumbel_input, self.noise], outputs=self.decode_out)
        self.model: keras.Model
        self.model.add_loss(custom_loss)
        self.model.add_metric(ber, name='ber')
        self.model.compile(optimizer=keras.optimizers.Adam(self.learning_rate))
        self.logits_model = keras.Model(
            inputs=self.snr_input, outputs=self.logits)
        self.tx_model = keras.Model(
            inputs=[self.snr_input, self.gumbel_input], outputs=self.tx_norm)

    def train_data_generator(self):
        snr_data = self.snr * \
            np.ones([self.train_batch_size, 1], dtype=np.float32)
        while True:
            gumbel_data = standard_gumbel(
                shape=[self.train_batch_size, self.M])
            noise = tf.random.normal(mean=0, stddev=self.noise_sigma, shape=[
                                     self.train_batch_size, 2])
            yield [snr_data, gumbel_data, noise]

    def train(self):
        # snr_data = self.snr * np.ones([self.train_batch_size * self.steps_per_epoch, 1], dtype=np.float32)
        # gumbel_data = standard_gumbel(shape=[self.train_batch_size * self.steps_per_epoch, self.M])
        # noise = tf.random.normal(mean=0, stddev=self.noise_sigma,
        #                          shape=[self.train_batch_size * self.steps_per_epoch, 2])
        # print(self.binary_sym)
        cbs = [
            keras.callbacks.EarlyStopping(
                monitor='loss', patience=2, verbose=1, mode='min')
        ]
        self.model.fit(self.train_data_generator(), batch_size=self.train_batch_size,
                       epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)
        # self.model.fit(x={'snr_input': snr_data, 'gumbel_inputs': gumbel_data, 'noise': noise},
        #                epochs=self.epochs, batch_size=self.train_batch_size)

    def cons_point_generate(self):
        snr_d = self.snr*np.ones([self.M, 1], dtype=np.float32)
        logits_out = self.logits_model(snr_d).numpy()
        gumbel_d = keras.utils.to_categorical(np.arange(self.M), self.M)
        self.sym_model = keras.Model(
            inputs=[self.snr_input, self.gumbel_input], outputs=self.sym)
        sym = self.sym_model([snr_d, gumbel_d])
        tx_norm = self.tx_model([snr_d, gumbel_d]).numpy()
        prob_s = tf.math.softmax(logits_out)
        prob_s = np.array(prob_s[0]).reshape([-1])
        s = prob_s*100
        print(logits_out)
        print(sym)
        plt.scatter(tx_norm[:, 0], tx_norm[:, 1], s=s)
        plt.show()
        plt.plot(prob_s)
        plt.show()


if __name__ == '__main__':
    comm.set_device(gpu_id=1, gpu_mem=4096)

    print("\n\ntf version:", tf.__version__)
    M_dic = {16, 64, 256}
    print("\n\n")
    M = int(input("modOrder(16,64,256):"))

    if M not in M_dic:
        raise ValueError("the modulation order should be 16, 64 or 256")

    start_snr = float(input("start snr:"))
    step_snr = float(input("step snr:"))
    end_snr = float(input("end snr:"))
    epochs = int(input("epochs:"))
    cce_weight = float(input("cce_weight:"))
    bce_weight = float(input("bce_weight:"))
    matfilename = input(".mat file's name:")

    st = datetime.datetime.now()
    log_path = "./log"
    log_name = "g{}.log".format(st.strftime("%m%d_%H%M%S"))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    comm.redirect2log(os.path.join(log_path, log_name))
    print("M={M}".format(M=M))
    print("cce_weight={}".format(cce_weight))
    print("bce_weight={}".format(bce_weight))

    if len(matfilename) <= 4 or matfilename[-4:] != ".mat":
        matfilename += ".mat"

    if os.path.exists(matfilename):
        copy_name = "copy_"+matfilename
        while os.path.exists(copy_name):
            copy_name = "copy_"+copy_name
        print("\n\n{name} has existed, it will be copy to {copy_name}, and then {name} will be recreated\n\n".format(
            name=matfilename, copy_name=copy_name))
        shutil.copy(matfilename, copy_name)
        os.remove(matfilename)

    snr = start_snr
    while snr <= end_snr:
        print("\n\nsnr={snr:.2f}".format(snr=snr))
        symwise_shaper = SymbolwiseShaping(
            M=M, snr=snr, epochs=epochs, cce_weight=cce_weight, bce_weight=bce_weight, is_ebn0=False)
        symwise_shaper.create_model()
        symwise_shaper.train()
        if matfilename != ".mat":
            symwise_shaper.evaluate(matfilename=matfilename)
        symwise_shaper.cons_prob_plot()
        snr += step_snr
    et = datetime.datetime.now()
    print("run time:", et-st)

    # symwise_shaper = SymbolwiseShaping(M=16,snr=6,epochs=10,cce_weight=0.5,bce_weight=1.0,is_ebn0=False)
    # symwise_shaper.create_model()
    # # symwise_shaper.model_whole.summary()
    # # symwise_shaper.train()
    # symwise_shaper.evaluate()
    # symwise_shaper.cons_prob_plot()

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
