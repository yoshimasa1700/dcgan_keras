import os
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from PIL import Image
from Generator import Generator
from Discriminator import Discriminator
import util
import lmdb
import cv2


class Trainer:
    BATCH_SIZE = 8
    NUM_EPOCH = 1000
    GENERATED_IMAGE_PATH = 'generated_images/'

    generator = Generator()
    discriminator = Discriminator()

    def train(self, path):
        env = lmdb.open(path, map_size=1099511627776,
                        max_readers=100, readonly=True)
        self.txn = env.begin(write=False)
        self.cursor = self.txn.cursor()

        for key, val in self.cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                np.fromstring(val, dtype=np.uint8), 1)
            cv2.imshow("sample", img)
            c = cv2.waitKey()
            if c == 27:
                break
        # load train datasets
        # (X_train, Y_train), (_, _) = cifar10.load_data()

        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = X_train.reshape(
        # X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)

        d_opt = Adam(lr=1e-5, beta_1=0.1)
        self.discriminator.model.compile(
            loss='binary_crossentropy', optimizer=d_opt)
        self.discriminator.model.trainable = False

        dcgan = Sequential([self.generator.model, self.discriminator.model])

        g_opt = Adam(lr=2e-4, beta_1=0.5)
        self.generator.model.compile(
            loss='binary_crossentropy', optimizer='Adam')
        dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

        # we should decide num batch fixd value
        # because resource problem.
        # num_batches = int(X_train.shape[0] / self.BATCH_SIZE)
        num_batches = 10

        for epoc in xrange(self.NUM_EPOCH):
            for batch_index in xrange(num_batches):
                noise = np.array([np.random.uniform(-1, 1, 100)
                                  for _ in range(self.BATCH_SIZE)])

                # TODO: extract batch from database.
                # image_batch = X_train[batch_index * self.BATCH_SIZE:
                #                       (batch_index + 1) * self.BATCH_SIZE]

                generated_images = self.generator.model.predict(
                    noise, verbose=0)

                if batch_index % 100 == 0:
                    if not os.path.exists(self.GENERATED_IMAGE_PATH):
                        os.mkdir(self.GENERATED_IMAGE_PATH)
                    util.save_images(
                        generated_images,
                        self.GENERATED_IMAGE_PATH
                        + "{:04d}_{:04d}.png".format(epoc, batch_index))

                X = np.concatenate((image_batch, generated_images))
                y = [1] * self.BATCH_SIZE + [0] * self.BATCH_SIZE

                self.discriminator.model.trainable = True
                d_loss = self.discriminator.model.train_on_batch(X, y)
                self.discriminator.model.trainable = False

                noise = np.array([np.random.uniform(-1, 1, 100)
                                  for _ in range(self.BATCH_SIZE)])
                g_loss = dcgan.train_on_batch(noise, [1] * self.BATCH_SIZE)

                print("epoch: {}, batch: {}, d_loss:{}, g_loss:{}".format(
                    epoc, batch_index, d_loss, g_loss))

            self.generator.model.save_weights('generator.h5')
            self.discriminator.model.save_weights('discriminator.h5')


if __name__ == "__main__":
    (X_train, Y_train), (_, _) = cifar10.load_data()
    print(X_train.shape)

    import cv2

    for image in X_train:
        cv2.imshow("test", image)
        cv2.waitKey(0)
