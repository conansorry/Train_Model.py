import numpy as np

import Info_GAN_Meta as GAN
import Train_Model as Train
import Prepare_Data as Data
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras_contrib
import Model_Pattern_To_Latten as PTL
import Model_Responce_To_Latten as RTL
import os
n_res = 32
n_pattern = (n_res, n_res, 1)
n_responce = (100)
p_lattent = 8
r_lattent = 50
Pattern, Responce = Data.load_real_samples()

## -------------------------------------------------

if( os.path.isfile("L2P.h5") and os.path.isfile("P2L.h5")):
    P2L_model = load_model("P2L.h5", custom_objects={'InstanceNormalization':InstanceNormalization})
    L2P_model = load_model("L2P.h5", custom_objects={'InstanceNormalization':InstanceNormalization})
else:
    P2L_model = PTL.define_Pattern_To_Latten( pattern_shape=(32,32,1), latten_dim = p_lattent,  n_resnet=3 )
    L2P_model = PTL.define_Latten_To_Pattern( latten_dim = p_lattent )
    PLP_model = PTL.define_Pattern_To_Latten_To_Pattern((32,32,1), P2L_model, L2P_model)

    Train.train_general_model(PLP_model, Pattern, n_epochs=1200)

    P2L_model.save("P2L.h5")
    L2P_model.save("L2P.h5")

## ----------------------------------------------------------


# -------------------------------------------------
if (os.path.isfile("L2R.h5") and os.path.isfile("R2L.h5")):
    R2L_model = load_model("R2L.h5")
    L2R_model = load_model("L2R.h5")
else:
    R2L_model = RTL.define_Responce_To_Lattern( responce_shape = (100), latten_dim = r_lattent,  n_resnet=3 )
    L2R_model = RTL.define_Lattern_To_Responce( latten_dim = r_lattent )
    RLR_model = RTL.define_Responce_To_Lattern_To_Responce( 100, R2L_model, L2R_model)

    Train.train_general_model(RLR_model, Responce, n_epochs=1200)

    R2L_model.save("R2L.h5")
    L2R_model.save("L2R.h5")
# ----------------------------------------------------------

d_model = GAN.define_discriminator(pattern_shape=n_pattern, pattern_latten_dim=p_lattent,
                                                         responce_shape=n_responce, responce_latten_dim=r_lattent)
g_model = GAN.define_generator(P2L_model, R2L_model)
gan_model = GAN.define_gan(g_model, d_model, pattern_shape=n_pattern, responce_shape=n_responce)
plot_model(gan_model, to_file='gan_model_plot.png', show_shapes=True,show_layer_names=True)
plot_model(g_model, to_file='generator_model_plot.png', show_shapes=True,show_layer_names=True)
plot_model(d_model, to_file='discriminator_model_plot.png', show_shapes=True,show_layer_names=True)
Train.train(g_model, d_model, gan_model, P2L_model, R2L_model, Pattern, Responce, n_epochs=1000)