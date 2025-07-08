import math

import numpy as np
import tensorflow as tf
from keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def Conv_block_mSEM(input_layer, F1=64, kernLength=64,  D=2, in_chans=32, dropout=0.3):
    F2 = 32
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    channel1 = [0, 15, 1, 14, 2, 11, 12, 3, 13]   # 9
    channel2 = [4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25]   # 13
    channel3 = [27, 26, 18, 29, 17, 30, 31, 16, 28, 19]    # 10

    # channel1
    block2_1 = tf.gather(block1, indices=channel1, axis=2)
    block2_1 = DepthwiseConv2D((1, 9), use_bias=False,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block2_1)
    # channel2
    block2_2 = tf.gather(block1, indices=channel2, axis=2)
    block2_2 = DepthwiseConv2D((1, 13), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_2)
    # channel3
    block2_3 = tf.gather(block1, indices=channel3, axis=2)
    block2_3 = DepthwiseConv2D((1, 10), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_3)

    block2 = tf.concat([block1, block2_1, block2_2, block2_3], axis=2)

    block2 = DepthwiseConv2D((1, in_chans + 3), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block2)

    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((4, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)  #  (none,)
    block3 = Dropout(dropout)(block3)
    return block3

def TCN_block(input_layer, input_dimension=32, depth=2, kernel_size=4, filters=32, dropout=0.3, activation='elu'):

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out

def UDCNN_BiTCN(n_classes, in_chans=32, in_samples=1500,
           eegn_F1=64, eegn_D=2, eegn_kernelSize=64,  eegn_dropout=0.3):

    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)
    block1 = Conv_block_mSEM(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    block3 = block1[:, ::-1, :]

    block2 = TCN_block(input_layer=block1, input_dimension=32, depth=2,
                       kernel_size=4, filters=32,
                       dropout=0.3, activation='elu')

    block2 = Lambda(lambda x: x[:, -1, :])(block2)

    block3 = TCN_block(input_layer=block3, input_dimension=32, depth=2,
                       kernel_size=4, filters=32,
                       dropout=0.3, activation='elu')
    block3 = Lambda(lambda x: x[:, -1, :])(block3)

    flatten1 = Flatten(name='flatten1')(block2)
    flatten2 = Flatten(name='flatten2')(block3)
    concatenated = Concatenate()([flatten1, flatten2])

    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(0.25))(concatenated)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_1, outputs=softmax)


