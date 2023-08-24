import Prepare_Data as Data
import numpy as np
from matplotlib import pyplot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import BatchNormalization
from keras.optimizers import Adam

def train_general_model( model_in, data_in, n_epochs=100, n_batch=128):
    # calculate the number of batches per training epoch
    bat_per_epo = int(data_in.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate
    for i in range(n_steps):
        p_real = Data.generate_unsorted_samples(data_in, n_batch)
        loss = model_in.train_on_batch(p_real, p_real)
        print('>%d, g[%.3f] ' % (i + 1, loss))
    summarize_general_performance(i + 1, model_in, p_real, n_samples= 20)


def train(g_model, d_model, gan_model, P2L_model, R2L_model, pattern, responce, n_epochs=100, n_batch=128, patch_shape=32, n_pre = 10000 ):

    # calculate the number of batches per training epoch
    bat_per_epo = int(pattern.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples

    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    # for layer in g_model.layers:
    #     layer.trainable = True
    g_model.compile(loss='mae', optimizer=Adam(lr=0.002))
    for i in range( n_pre ):
        p_real, r_real, y_real = Data.generate_real_samples(pattern, responce, n_batch)

        p_lattent = Data.generate_latent_samples(p_real, n_batch, weight=0.9)
        # p_lattent = Data.generate_unsorted_samples(pattern, n_batch)
        # g_loss_pre = g_model.train_on_batch([p_real, r_real], p_real)
        g_loss_pre = g_model.train_on_batch([p_lattent, r_real], p_real)

        print('>%d, g[%.3f] ' % (i + 1, g_loss_pre))
    summarize_performance(i, g_model, d_model, gan_model, p_real, r_real, n_samples=10)
    summarize_performance_a(i+1, g_model, d_model, gan_model, p_real, r_real, n_samples=10)

    # for i in range(2):
    for i in range(n_steps):
        # for layer in d_model.layers:
        #     layer.trainable = True
        # g_model.compile(loss='mae', optimizer=Adam(lr=0.002))
        # gan_model.compile(loss=['binary_crossentropy', "mae"], optimizer=Adam(lr=0.002), loss_weights=[1.0, 0.1])
        # d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002))
        # print("---------------dddddd---------------")
        # d_model.summary()
        # g_model.summary()
        # P2L_model.summary()
        # R2L_model.summary()
        # print("finished dddddd")
        # get randomly selected 'real' samples
        p_real, r_real, y_real = Data.generate_real_samples(pattern, responce, half_batch)
        # update discriminator and q model weights
        d_loss1 = d_model.train_on_batch([p_real, r_real], y_real)
        # generate 'fake' examples
        p_lattent = Data.generate_unsorted_samples(p_real, half_batch)

        p_fake, y_fake = Data.generate_fake_samples(g_model, p_lattent, r_real, half_batch, patch_shape)
        # update discriminator model weights

        d_loss2 = d_model.train_on_batch([p_fake, r_real], y_fake)

        # prepare points in latent space as input for the generator

        p_tar_gan, r_gan, _ = Data.generate_real_samples(pattern, responce, n_batch)
        p_gan = Data.generate_unsorted_samples(p_tar_gan, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the g via the d and q error
        # for layer in d_model.layers:
        #     layer.trainable = False
        #     if isinstance(layer, BatchNormalization):
        #         layer.trainable = True
        #     if isinstance(layer, InstanceNormalization):
        #         layer.trainable = True
        # g_model.compile(loss='mae', optimizer=Adam(lr=0.002))
        # gan_model.compile(loss=['binary_crossentropy', "mae"], optimizer=Adam(lr=0.002), loss_weights=[1.0, 0.1])
        # d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002))
        # print("---------------gggggggg---------------")
        #
        # P2L_model.summary()
        # R2L_model.summary()
        # g_model.summary()
        # d_model.summary()
        # gan_model.summary()
        #
        # print("finished gfggggggg")

        # exit()
        g_loss, g_d, g_mat = gan_model.train_on_batch([p_gan, r_gan], [y_gan, p_tar_gan])
        # summarize loss on this batch
        print('>%d, d[%.3f,%.3f], g[%.3f, %.3f, %.3f] ' % (i+1, d_loss1, d_loss2, g_loss, g_d, g_mat ))
        # evaluate the model performance every 'epoch'
        if (i+1) % (bat_per_epo * 50) == 0:
            summarize_performance(i, g_model, d_model, gan_model, p_gan, r_real, n_samples = 10)




def summarize_performance(step, g_model, d_model, gan_model, p_real, r_real, n_samples=10):
    # prepare fake examples
    Try_Pattern = Data.generate_unsorted_samples(p_real, n_samples)
    X, _ = Data.generate_fake_samples(g_model, Try_Pattern, r_real, n_samples, patch_shape=32)
    # plot images

    for i in range(10):
    # define subplot
        pyplot.subplot(5, 8, 1 + 4*i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data


        img = np.zeros( shape = (p_real.shape[1], p_real.shape[2], 1) )
        img[:,:,0] = X[i, :, :, 0]
        pyplot.imshow(img[:, :, :])

        pyplot.subplot(5, 8, 2 + 4*i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        img[:,:,0] = Try_Pattern[i, :, :, 0]
        pyplot.imshow(img[:, :, :])
        # save plot to file

        pyplot.subplot(5, 8, 3 + 4 * i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        img[:, :, 0] = p_real[i, :, :, 0]
        pyplot.imshow(img[:, :, :])
        # save plot to file

        pyplot.subplot(5, 8, 4 + 4 * i)
        # plot raw pixel data
        pyplot.axis('off')
        pyplot.plot(r_real[i, :])


    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)

    filename3 = 'd_model_%04d.h5' % (step + 1)
    d_model.save(filename3)
    # save the gan model
    filename4 = 'gan_model_%04d.h5' % (step+1)
    gan_model.save(filename4)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

def summarize_general_performance(step, g_model, pattern, n_samples=20):
    # prepare fake examples
    Try_Pattern = Data.generate_unsorted_samples(pattern, n_samples)
    X = g_model.predict(Try_Pattern)
    # plot images
    n_col = 8
    n_row = int(2 * n_samples/n_col)
    # fig, ax = pyplot.subplots(nrows = n_row, ncols = n_col )
    for i in range(n_samples):
        if( Try_Pattern.ndim == 4 ):
    # define subplot
            pyplot.subplot(n_row, n_col, 1 + 2*i, )
            # turn off axis
            pyplot.axis('off')
            pyplot.imshow(X[i,:, :, :])

            pyplot.subplot(n_row, n_col, 2 + 2*i)
            pyplot.axis('off')
            pyplot.imshow(Try_Pattern[i,:, :, :])
        elif (Try_Pattern.ndim == 2):
            pyplot.subplot(n_row, n_col, 1 + 2 * i, )
            # turn off axis
            pyplot.axis('off')
            # pyplot.axis(share
            pyplot.ylim(top=1,bottom=0)
            pyplot.plot(X[i, :])
            pyplot.plot(Try_Pattern[i, :])

            pyplot.subplot(n_row, n_col, 2 + 2 * i)
            pyplot.axis('off')
            pyplot.ylim(top=1, bottom=0)
            pyplot.plot(np.abs(X[i, :]-Try_Pattern[i, :]))

    # fig.tight_layout()

        # save plot to file


    # save plot to file
    filename1 = 'tunemodel_plot_%04d.png' % (step+1)
    pyplot.tight_layout()
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)

    print('>Saved: %s and %s' % (filename1, filename2))




def summarize_performance_a(step, g_model, d_model, gan_model, p_real, r_real, n_samples=10):
    # prepare fake examples
    Try_Pattern = p_real[0:n_samples]
    X, _ = Data.generate_fake_samples(g_model, p_real[0:n_samples], r_real[0:n_samples], n_samples, patch_shape=32)
    # plot images

    for i in range(10):
    # define subplot
        pyplot.subplot(5, 8, 1 + 4*i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data


        img = np.zeros( shape = (p_real.shape[1], p_real.shape[2], 1) )
        img[:,:,0] = X[i, :, :, 0]
        pyplot.imshow(img[:, :, :])

        pyplot.subplot(5, 8, 2 + 4*i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        img[:,:,0] = Try_Pattern[i, :, :, 0]
        pyplot.imshow(img[:, :, :])
        # save plot to file

        pyplot.subplot(5, 8, 3 + 4 * i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        img[:, :, 0] = p_real[i, :, :, 0]
        pyplot.imshow(img[:, :, :])
        # save plot to file

        pyplot.subplot(5, 8, 4 + 4 * i)
        # plot raw pixel data
        pyplot.axis('off')
        pyplot.plot(r_real[i, :])


    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
