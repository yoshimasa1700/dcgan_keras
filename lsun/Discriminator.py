from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D


class Discriminator:
    def __init__(self):
        model = Sequential()
        model.add(Conv2D(256, (5, 5),
                   padding='same',
                   input_shape=(64, 64, 3)))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128, (5, 5)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        self.model = model

if __name__ == "__main__":
    discriminator = Discriminator()
