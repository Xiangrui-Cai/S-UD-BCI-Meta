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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def Conv_block_mSEM(input_layer, F1=64, kernLength=64,  D=2, in_chans=32, dropout=0.3):
    F2 = 32
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    # channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'Cz', 'FC2',
    #         'FC6', 'F8', 'F4', 'Fz', 'AF4', 'Fp2', 'O2', 'PO4', 'P4', 'P8',
    #         'CP6', 'CP2', 'C4', 'T8', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'PO3',
    #         'O1', 'Oz']

    channel1 = [0, 15, 1, 14, 2, 11, 12, 3, 13]   # 9
    channel2 = [27, 26, 18, 29, 17, 30, 31, 16, 28, 19]    # 10
    channel3 = [4, 9]   # 2
    channel4 = [7, 8, 22]   # 3
    channel5 = [21, 25]   # 2
    channel6 = [5, 6, 24]   # 3
    channel7 = [10, 20, 23]   # 3

    # channel1
    block2_1 = tf.gather(block1, indices=channel1, axis=2)
    block2_1 = DepthwiseConv2D((1, 9), use_bias=False,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block2_1)
    # channel2
    block2_2 = tf.gather(block1, indices=channel2, axis=2)
    block2_2 = DepthwiseConv2D((1, 10), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_2)
    # channel3
    block2_3 = tf.gather(block1, indices=channel3, axis=2)
    block2_3 = DepthwiseConv2D((1, 2), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_3)
    
    # channel4
    block2_4 = tf.gather(block1, indices=channel4, axis=2)
    block2_4 = DepthwiseConv2D((1, 3), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_4)
    
    # channel5
    block2_5 = tf.gather(block1, indices=channel5, axis=2)
    block2_5 = DepthwiseConv2D((1, 2), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_5)
    
    # channel6
    block2_6 = tf.gather(block1, indices=channel6, axis=2)
    block2_6 = DepthwiseConv2D((1, 3), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_6)
    
    # channel7
    block2_7 = tf.gather(block1, indices=channel7, axis=2)
    block2_7 = DepthwiseConv2D((1, 3), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_7)

    block2 = tf.concat([block1, block2_1, block2_2, block2_3, block2_4, block2_5, block2_6, block2_7], axis=2)

    block2 = DepthwiseConv2D((1, in_chans + 7), use_bias=False,
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


def Conv_block_mSEM_3(input_layer, F1=64, kernLength=64, D=2, in_chans=32, dropout=0.3):
    F2 = 32
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    # channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'Cz', 'FC2',
    #         'FC6', 'F8', 'F4', 'Fz', 'AF4', 'Fp2', 'O2', 'PO4', 'P4', 'P8',
    #         'CP6', 'CP2', 'C4', 'T8', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'PO3',
    #         'O1', 'Oz']

    channel1 = [0, 15, 1, 14, 2, 11, 12, 3, 13]  # 9
    channel2 = [27, 26, 18, 29, 17, 30, 31, 16, 28, 19]  # 10
    channel3 = [4, 9, 7, 8, 22, 21, 25, 5, 6, 24, 10, 20, 23]  # 2

    # channel1
    block2_1 = tf.gather(block1, indices=channel1, axis=2)
    block2_1 = DepthwiseConv2D((1, 9), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_1)
    # channel2
    block2_2 = tf.gather(block1, indices=channel2, axis=2)
    block2_2 = DepthwiseConv2D((1, 10), use_bias=False,
                               data_format='channels_last',
                               depthwise_constraint=max_norm(1.))(block2_2)
    # channel3
    block2_3 = tf.gather(block1, indices=channel3, axis=2)
    block2_3 = DepthwiseConv2D((1, 2), use_bias=False,
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
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)  # (none,)
    block3 = Dropout(dropout)(block3)
    return block3

def TCN_block(input_layer, input_dimension=32, depth=2, kernel_size=4, filters=32, dropout=0.3, activation='elu'):

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    # print(block.shape)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    # print(block.shape)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)
    # print(out.shape)

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
        # print(block.shape)
        added = Add()([block, out])
        out = Activation(activation)(added)
        # print(out.shape)

    return out

def UDCNN_BiTCN(n_classes, in_chans=32, in_samples=1500,
           eegn_F1=64, eegn_D=2, eegn_kernelSize=64,  eegn_dropout=0.3):

    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)
    block1 = Conv_block_mSEM(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    # Bi
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

    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(0.3))(concatenated)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_1, outputs=softmax)



def main():
    print("Starting test...")

    # 参数设置
    n_classes = 4       # 分类数量
    in_chans = 32       # 输入通道数
    in_samples = 1500   # 时间点数
    batch_size = 32
    num_samples = 100   # 训练样本数
    epochs = 5

    # 构建模型
    print("Building model...")
    model = UDCNN_BiTCN(n_classes=n_classes, in_chans=in_chans, in_samples=in_samples)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 打印模型结构
    model.summary()

    # 生成随机 EEG 数据用于测试
    print(f"Generating {num_samples} random samples...")
    X_train = np.random.rand(num_samples, 1, in_chans, in_samples).astype(np.float32)
    y_train = np.random.randint(low=0, high=n_classes, size=(num_samples,))
    y_train_cat = to_categorical(y_train, num_classes=n_classes)

    # 验证集
    X_val = np.random.rand(20, 1, in_chans, in_samples).astype(np.float32)
    y_val = np.random.randint(low=0, high=n_classes, size=(20,))
    y_val_cat = to_categorical(y_val, num_classes=n_classes)

    # 开始训练
    print("Training model...")
    history = model.fit(X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    # 输出结果
    print("Training completed.")
    print("Final training accuracy:", history.history['accuracy'][-1])
    print("Final validation accuracy:", history.history['val_accuracy'][-1])

if __name__ == '__main__':
    main()


