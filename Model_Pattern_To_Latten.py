from keras.models import Input, Model
from keras.layers import LeakyReLU, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.layers import Activation, Concatenate, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model



def resnet_cnn_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])

    return g

def define_Pattern_To_Latten(pattern_shape=(32,32,1), latten_dim = 80,  n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(pattern_shape))
    # c7s1-64
    g = Conv2D(32, (7, 7), padding='same', kernel_initializer = init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # d256
    g = Conv2D(32, (3, 3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    for _ in range(n_resnet):
        g = resnet_cnn_block(32, g)

    # d512
    g = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # Flatten
    g = Flatten()(g)

    g = Dense(128, activation='relu')(g)
    g = Dense(64, activation='relu')(g)
    g = Dense(latten_dim, activation='sigmoid')(g)

    model = Model(in_image,g)
    return model

def define_Latten_To_Pattern( latten_dim ):
    init = RandomNormal(stddev=0.02)

    # pattern_input = Input(shape=(pattern_shape))
    Input_L = Input(shape=(latten_dim,))

    m = Dense(64, activation='relu')(Input_L)
    m = Dense(128, activation='relu')(m)
    m = Dense(1024, activation='relu')(m)
    m = Dense(4096, activation='relu')(m)

    m = Reshape(target_shape=(8,8,64))(m)

    gen = Conv2D(64, (4, 4), padding='same', kernel_initializer=init)(m)
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
    model = Model(Input_L, out_layer)

    model.compile( optimizer=Adam(lr=0.002), loss="mae", )
    return model

def define_Pattern_To_Latten_To_Pattern( pattern_shape, model_P2L, model_L2P ):

    pattern = Input(shape=(pattern_shape))
    lattent = model_P2L(pattern)
    output = model_L2P(lattent)
    model = Model(pattern, output)

    model.compile( optimizer=Adam(lr=0.002), loss="mae", )
    return model

#
# image_shape = (128,128,2)
# # create the model
# # model = define_discriminator(image_shape)
# model = define_Pattern_To_Latten(image_shape)
# # summarize the model
# model.summary()
#
# plot_model(model, to_file='generator_model_plot.png', show_shapes=True,show_layer_names=True)