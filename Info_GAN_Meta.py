from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.layers import Concatenate, Flatten, BatchNormalization
from keras.layers import Activation, Conv2D, Conv2DTranspose, Reshape
from keras.models import Model
from keras.optimizers import Adam
from Model_Pattern_To_Latten import define_Pattern_To_Latten
from Model_Responce_To_Latten import define_Responce_To_Lattern
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def define_discriminator(pattern_shape=(32,32,1), pattern_latten_dim=80,
                         responce_shape=(100), responce_latten_dim=40):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    model_P2L = define_Pattern_To_Latten(pattern_shape, pattern_latten_dim, n_resnet=3)
    model_R2L = define_Responce_To_Lattern(responce_shape, responce_latten_dim, n_resnet=3)

    d = Concatenate()([model_P2L.output, model_R2L.output])

    d = Dense(64, activation="relu", kernel_initializer=init)(d)
    d = Dropout(rate=0.05)(d)
    d = Dense(64, activation="relu", kernel_initializer=init)(d)
    d = Dropout(rate=0.05)(d)
    d = Dense(64, activation="relu", kernel_initializer=init)(d)
    d = Dropout(rate=0.05)(d)
    # d = Dense(64, activation="relu", kernel_initializer=init)(d)
    # d = Dropout(rate=0.05)(d)
    # d = Dense(64, activation="relu", kernel_initializer=init)(d)
    # d = Dropout(rate=0.05)(d)
    # d = Dense(64, activation="relu", kernel_initializer=init)(d)
    # d = Dropout(rate=0.05)(d)
    # d = Dense(64, activation="relu", kernel_initializer=init)(d)
    # d = Dropout(rate=0.05)(d)
    d = Dense(32, activation="relu", kernel_initializer=init)(d)
    d = Dense(16, activation="relu", kernel_initializer=init)(d)
    d = Dense(8, activation="relu", kernel_initializer=init)(d)
    patch_out = Dense(1, activation="sigmoid")(d)
    # patch_out = LeakyReLU(alpha=0.2)(patch_out)
    model = Model([model_P2L.input, model_R2L.input], patch_out)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002))
    return model, model_P2L, model_R2L



def define_generator( model_P2L, model_R2L ):

    # for layer in model_P2L.layers:
    #     if not isinstance(layer, InstanceNormalization):
    #         layer.trainable = False
    # for layer in model_R2L.layers:
    #     if not isinstance(layer, BatchNormalization):
    #         layer.trainable = False

    init = RandomNormal(stddev = 0.02)

    pattern_lattern = model_P2L.output
    responce_lattern = model_R2L.output

    # pattern_lattern = Dropout(rate=0.5)(pattern_lattern)
    gen = Concatenate()([pattern_lattern, responce_lattern])

    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes, kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    gen = Reshape((8, 8, 128))(gen)
    # normal
    gen = Conv2D(64, (4, 4), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    # upsample to 16x16
    gen = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    # upsample to 32x32
    gen = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)

    # normal
    gen = Conv2D(32, (2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)

    # normal
    gen = Conv2D(1, (2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)

    # sigmoid output
    out_layer = Activation('sigmoid')(gen)
    # define model
    model = Model([model_P2L.input, model_R2L.input], out_layer)
    return model

def define_gan(g_model, d_model,
               pattern_shape = (32,32,1), responce_shape = (100)):



    pattern_input = Input(shape=(pattern_shape))
    responce_input = Input(shape=(responce_shape,))


    gen_out = g_model([pattern_input, responce_input])

    d_output = d_model([gen_out, responce_input])

    model = Model([pattern_input, responce_input], [d_output,gen_out] )
    opt = Adam(lr=0.002)
    model.compile(loss=['binary_crossentropy',"mae"], optimizer = opt,loss_weights=[1.0, 0.1])
    # for layer in d_model.layers:
    #     layer.trainable = False
    #     if isinstance(layer, BatchNormalization):
    #         layer.trainable = True
    #     if isinstance(layer, InstanceNormalization):
    #         layer.trainable = True
    return model
