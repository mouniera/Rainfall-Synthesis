import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy.typing import NDArray
import numpy as np

#############################################################
#CAE code for dimension reduction 
#inspired by : https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
#############################################################

def CAE(input_shape=(1,256, 384), filters=[16, 32, 64, 128], latent_space=40, strides=[4,4,2,2], size_filters=[32,16,3,3]):
    """ Autoencoder architecture
    Args:
        input_shape (tuple, optional): input shape. Defaults to (1,256, 384).
        filters (list, optional): number of filters for the different layers. Defaults to [16, 32, 64, 128, 10].
        latent_space (int, optional): Size of the latent space. Defaults to 40.
        strides (list, optional): strides for the different layers. Defaults to [4,4,2,2].
        size_filters (list, optional): size of filters for the different layers. Defaults to [32,16,3,3].

    Returns:
        tf.keras.Model : Autoencoder model 
    """
    model = Sequential()

    model.add(Conv2D(filters[0], size_filters[0], strides=strides[0], padding='same', activation='relu', name='conv1', input_shape=input_shape, data_format='channels_first'))
    model.add(Conv2D(filters[1], size_filters[1], strides=strides[1], padding='same', activation='relu', name='conv2', data_format='channels_first'))
    model.add(Conv2D(filters[2], size_filters[2], strides=strides[2], padding='same', activation='relu', name='conv3', data_format='channels_first'))
    model.add(Conv2D(filters[3], size_filters[3], strides=strides[3], padding='same', activation='relu', name='conv4', data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(units=latent_space, name='embedding'))
    reduction=np.prod(strides)
    model.add(Dense(units=filters[3]*int(input_shape[1]/reduction)*int(input_shape[2]/reduction), activation='relu'))
    model.add(Reshape((filters[3],int(input_shape[1]/reduction), int(input_shape[2]/reduction))))

    model.add(Conv2DTranspose(filters[2], size_filters[3], strides=strides[3], padding='same', activation='relu', name='deconv4', data_format='channels_first'))
    model.add(Conv2DTranspose(filters[1], size_filters[2], strides=strides[2], padding='same', activation='relu', name='deconv3', data_format='channels_first'))
    model.add(Conv2DTranspose(filters[0], size_filters[1], strides=strides[1], padding='same', activation='relu', name='deconv2', data_format='channels_first'))
    model.add(Conv2DTranspose(input_shape[0], size_filters[0], strides=strides[0], padding='same', name='deconv1', data_format='channels_first'))
    model.summary()
    return model


def Run(data_RR: NDArray[np.float32],Nb_filters: list[int],latent_space: int,size_filters: list[int],pmin: int,m: int,s: int,n: int,strides=[4,4,2,2],batch_size=256,l_rate=0.001,epochs_max=80) :
    """ Run autoencoder training

    Args:
        data_RR (NDArray[np.float32]): rainfall dataset (after importance sampling)
        Nb_filters (list[int]): number of filters for the different layers
        latent_space (int): latent space size
        size_filters (list[int]): size of filters for the differents layers
        pmin (int): importance sampling hyperparameter. Minimum probability of saving a sample
        m (int): _importance sampling hyperparameter. Multiplying factor.
        s (int): importance sampling hyperparameter. Rainfall interest threshold.
        n (int): Number of attempt
        strides (list, optional): . Defaults to [4,4,2,2].
        batch_size (int, optional): Batch size for training. Defaults to 256.
        l_rate (float, optional): learning rate for training. Defaults to 0.001.
        epochs_max (int, optional): epoch maximum for training. Defaults to 80.
    """

    filters_str= '-'.join(str(e) for e in Nb_filters)
    size_filters_str= '-'.join(str(e) for e in size_filters)
    strides_str= ''.join(str(e) for e in strides)

    model= CAE(filters=Nb_filters, strides=strides, latent_space=latent_space, size_filters=size_filters)
    name_config="RR1h_4year_importance_pmin"+str(pmin)+"_m"+str(m)+"_s"+str(s)+"_Strides"+strides_str+"_Filters_nb_"+filters_str+"_size_"+size_filters_str+"_Latent_size_"+str(latent_space)+"_Batch_"+str(batch_size)+"_Epochs_"+str(epochs_max)+"_lr"+str(l_rate)+"_n"+str(n)
    json_string = model.to_json()
    open(name_config +'_architecture.json', 'w').write(json_string)

    opt = tf.keras.optimizers.Adam(learning_rate=l_rate)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    model.compile(optimizer=opt, loss='mse', experimental_run_tf_function=False)    
    checkpointer = ModelCheckpoint(filepath=name_config+'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
    model.fit(data_RR, data_RR, batch_size=batch_size, epochs=epochs_max, shuffle=True, validation_split=0.2, callbacks= [checkpointer])
    return "OK"
     
