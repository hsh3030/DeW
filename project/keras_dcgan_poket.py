import os.path
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')

class Gan:
    def __init__(self, img_data):
        img_size = img_data.shape[1]
        channel = img_data.shape[3] if len(img_data.shape) >= 4 else 1

        self.img_data = img_data
        self.input_shape = (img_size, img_size, channel)

        self.img_rows = img_size
        self.img_cols = img_size
        self.channel = channel
        self.noise_size = 100

        # Creat D and G
        self.create_d()
        self.create_g()

        # Build model to train G
        optimizer = Adam(lr=0.0004)
        self.D.trainable = False
        self.AM = Sequential()
        self.AM.add(self.G)
        self.AM.add(self.D)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer)

    def create_d(self):
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*8, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def create_g(self):
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 8
        self.G.add(Dense(dim*dim*depth, imput_dim = self.noise_size))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())self.G.add(UpSampling2D())self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def train(self, batch_size=100):
        # Pick image data randomly
        images_train = self.img_data[np.random.randint(0, self.img_data.shape[0], size = batch_size), :, :, :]

        # Generate images from noise
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_size])
        images_fake = self.G.predict(noise)

        # Train D
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        self.D.trainable = True
        d_loss = self.D.train_on_batch(x, y)

        # Train G
        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_size])
        self.D.trainable = False
        a_loss = self.AM.train_on_batch(noise, y)

        return d_loss, a_loss, images_fake

    def save(self):
        self.G.save_weights('gan_g_weights.h5')
        self.D.save_weights('gan_d_weights.h5')

    def load(self):
        if os.path.isfile('gam_g_weights.h5'):
            self.G.load_weights('gan_g_weights.h5')
            print("Load G from file.")
