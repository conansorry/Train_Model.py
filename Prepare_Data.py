import numpy as np
import os

def load_real_samples():

    cur_path = "D:\Dropbox\Current_Research\\1 Metasurface GAN\Training data\Patterned_Pixcel"


    Pattern_path = "Mat"
    Pattern = []
    n_t = 2083

    for i in range(0, n_t):
        pattern_name = "Mat_design" + str(i) + ".txt"
        load_path = os.path.join(cur_path, Pattern_path, pattern_name)
        if( i % int(n_t/10) == 0 ):
            print(load_path)
        pattern_mat = np.loadtxt(load_path, delimiter=',')
        T_pattern = np.zeros((32,32,1))
        T_pattern[:,:,0] = pattern_mat[:,:]
        Pattern.append(T_pattern)

    Responce_path = "Res"
    Responce = []

    for i in range(0, n_t):
        Responce_name = "design" + str(i) + ".txt"
        load_path = os.path.join(cur_path, Responce_path, Responce_name)
        if( i % int(n_t/10) == 0 ):
            print(load_path)
        responce_mat = np.loadtxt(load_path, skiprows=8)
        resp = np.zeros((100))
        # print(responce_mat)
        for ii in range(responce_mat.shape[0]):
            resp[ii] = np.sqrt( responce_mat[ii,1]*responce_mat[ii,1] + responce_mat[ii,2]*responce_mat[ii,2] )

        Responce.append(resp)

    Pattern = np.asarray(Pattern)
    Responce = np.asarray(Responce)
    # print(Pattern.shape[:], Responce.shape[:])\
    print(Responce)
    return Pattern, Responce
# load_real_samples()

def generate_real_samples(pattern, responce, n_samples):

    # choose random instances
    ix = np.random.randint(0, pattern.shape[0], n_samples)
    # retrieve selected images
    P1, R1 = pattern[ix], responce[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return P1, R1, y

# generate points in latent space as input for the generator
def generate_latent_points(patch_shape, n_samples):
    # generate points in the latent space
    z_latent = np.random.randint(0,1,size = patch_shape * patch_shape * n_samples)
    # reshape into a batch of inputs for the network
    z_latent = z_latent.reshape(n_samples, patch_shape, patch_shape,1)

    return z_latent


def generate_latent_samples(pattern, n_samples, weight=0.5):
    # generate points in the latent space
    z_latent = generate_latent_points(pattern.shape[1], n_samples)

    # reshape into a batch of inputs for the network
    z_latent = z_latent.reshape(n_samples, pattern.shape[1], pattern.shape[1],1)
    weight_samples = weight * z_latent + (1.0 - weight) * pattern

    return weight_samples

def generate_unsorted_samples( pattern, n_samples ):
    ix = np.random.randint(0, pattern.shape[0], n_samples)
    P1 = pattern[ix]
    return P1

def generate_fake_samples(g_model, samples, r_real, n_samples, patch_shape, weight=0.1):
    # generate fake instance
    z_latent = generate_latent_points(patch_shape, n_samples)
    weight_samples = weight * z_latent + (1.0-weight) * samples
    X = g_model.predict([weight_samples, r_real])
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y