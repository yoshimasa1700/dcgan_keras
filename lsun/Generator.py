import numpy as np
import util

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D


class Generator:
    def __init__(self):

        model = Sequential()
        model.add(Dense(input_dim=100, units=512 * 16 * 16))
        model.add(Reshape((16, 16, 512), input_shape=(512*16*16,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(3, (5, 5), padding='same'))
        model.add(Activation('tanh'))

        self.model = model


    def load_model(self, path):
        self.model.load_weights(path)


    def generate(self):
        noise = np.array([np.random.uniform(-1, 1, 512)
                          for _ in range(10)])
        generated_images = self.model.predict(noise, verbose=0)
        util.save_images(generated_images, "generated_image.png")


if __name__ == "__main__":
    generator = Generator()
