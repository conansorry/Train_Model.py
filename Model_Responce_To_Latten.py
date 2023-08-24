from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.layers import BatchNormalization, Concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam

def resnet_dnn_block( n_filters, input_layer ):
    ini = RandomNormal(stddev=0.02)
    # first dense layer
    res = Dense( n_filters, kernel_initializer= ini ) (input_layer)
    res = BatchNormalization()(res)
    res = LeakyReLU(alpha = 0.05)(res)
    # second dense layer
    res = Dense( n_filters, kernel_initializer= ini )(res)
    res = Dropout( rate = 0.05 )(res)
    # concatenate merge channel-wise with input layer
    res = Concatenate()([res, input_layer])

    return res

def define_Responce_To_Lattern( responce_shape = (100), latten_dim = 40, n_resnet = 10 ):

    init = RandomNormal( stddev=0.02 )

    in_Responce = Input( shape =(responce_shape,))

    # P2L = Flatten()(define_Responce_To_Lattern)
    P2L = Dense( latten_dim, kernel_initializer=init)(in_Responce)
    P2L = LeakyReLU(alpha=0.05)(P2L)

    for _ in range(n_resnet):
        P2L = resnet_dnn_block( latten_dim, P2L )

    for _ in range(3):
        P2L = Dense(latten_dim, kernel_initializer=init)(P2L)
        P2L = LeakyReLU(alpha=0.05)(P2L)


    P2L = Dense( latten_dim, activation = 'relu' )(P2L)
    P2L = Dense( latten_dim, activation='sigmoid')(P2L)

    model = Model(in_Responce, P2L)
    return model


def define_Lattern_To_Responce( latten_dim ):
    init = RandomNormal(stddev=0.02)

    Input_layer = Input((latten_dim,))

    m = Dense(latten_dim, activation = 'relu')(Input_layer)
    m = Dense(latten_dim, activation = 'relu')(m)

    for _ in range(10):
        m = Dense(2*latten_dim)(m)
        m = LeakyReLU(alpha=0.05)(m)

    Output_layer  = Dense(100, activation="sigmoid")(m)

    model = Model(Input_layer, Output_layer)
    model.compile( optimizer=Adam(lr=0.002),loss="mae")
    return model




def define_Responce_To_Lattern_To_Responce( responce_shape, model_R2L, model_L2R ):
    pattern = Input(shape=(responce_shape,))
    lattent = model_R2L(pattern)
    output = model_L2R(lattent)
    model = Model(pattern, output)

    model.compile(optimizer=Adam(lr=0.002), loss="mae")
    return model




