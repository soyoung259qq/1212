from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate


def create_conv_block(inputs, filter_num):

    conv = Conv2D(filter_num, (3, 3), padding='same')(inputs)
    bn = Activation('relu')(conv)
    conv = Conv2D(filter_num, (3, 3), padding='same')(bn)
    bn = BatchNormalization(axis=3)(conv)
    bn = Activation('relu')(bn)
    pool = MaxPooling2D(pool_size=(2, 2))(bn)
    return pool, conv, bn


def create_concat_block(input_bn, input_conv, filter_num):

    up = concatenate([Conv2DTranspose(filter_num, (2, 2), strides=(2, 2), padding='same')(input_bn), input_conv], axis=3)
    conv = Conv2D(512, (3, 3), padding='same')(up)
    bn = Activation('relu')(conv)
    conv = Conv2D(512, (3, 3), padding='same')(bn)
    bn = BatchNormalization(axis=3)(conv)
    bn = Activation('relu')(bn)
    return bn


def create_model(input_size=(256, 256, 3)):

    inputs = Input(input_size)
    pool1, conv1, bn1 = create_conv_block(inputs, 64)
    pool2, conv2, bn2 = create_conv_block(pool1, 128)
    pool3, conv3, bn3 = create_conv_block(pool2, 256)
    pool4, conv4, bn4 = create_conv_block(pool3, 512)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    bn6 = create_concat_block(bn5, conv4, 512)
    bn7 = create_concat_block(bn6, conv3, 256)
    bn8 = create_concat_block(bn7, conv2, 128)
    bn9 = create_concat_block(bn8, conv1, 64)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])