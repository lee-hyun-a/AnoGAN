import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import cv2 as cv
from skimage.transform import resize
from skimage import img_as_ubyte
import glob
import shutil
import librosa
from scipy.stats import norm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten, Activation
from tensorflow.keras import backend as K

###############################################################
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
###############################################################

noise_dim=32

class DataLoader:
    def __init__(self, raw_path, train_path, test_path):
        self.raw_path = raw_path
        self.train_path = train_path
        self.test_path = test_path

    def split_train_test(self):
        allpath = glob.glob(self.raw_path + '*')
        for i, p in enumerate(allpath):

            librosa_wav, librosa_sr = librosa.load(p, sr=256, duration=1)

            print(i, p, np.max(librosa_wav), np.min(librosa_wav))

            os.makedirs(self.train_path, exist_ok=True)
            os.makedirs(self.test_path, exist_ok=True)

            if np.random.rand() <= 0.95:
                np.save(self.train_path + '{}'.format(i), np.expand_dims(librosa_wav, axis=1))
            else:
                np.save(self.test_path + '{}'.format(i), np.expand_dims(librosa_wav, axis=1))

    def map_func(self, feature_path):
        feature = np.load(feature_path).astype(np.float32)
        return feature

    def make_dataset(self, path, BATCH_SIZE):
        allpath = glob.glob(path + '*')
        dataset = tf.data.Dataset.from_tensor_slices(allpath)

        dataset = dataset.map(lambda item: tf.numpy_function(
            self.map_func, [item], tf.float32),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)

        return dataset


def conv_block(x,
               filters,
               activation,
               kernel_size=9,
               strides=1,
               padding="same",
               use_bias=True,
               use_bn=False,
               use_dropout=False,
               drop_value=0.5,
               training=True
               ):
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x, training=training)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model(training=True):
    img_input = layers.Input(shape=(256, 1))

    x = conv_block(
        img_input,
        32,
        kernel_size=25,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
        training=training
    )

    x = conv_block(
        x,
        64,
        kernel_size=16,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
        training=training
    )

    x = conv_block(
        x,
        128,
        kernel_size=9,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
        training=training
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation=layers.LeakyReLU(0.2))(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


def upsample_block(
        x,
        filters,
        activation,
        kernel_size=16,
        strides=1,
        up_size=2,
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
        training=True
):
    x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x, training=training)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_generator_model(training=True):
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(32 * 64, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((32, 64))(x)

    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
        kernel_size=9,
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
        training=training
    )

    x = upsample_block(
        x,
        32,
        layers.LeakyReLU(0.2),
        kernel_size=16,
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
        training=training
    )

    x = upsample_block(
        x, 1, layers.Activation("tanh"), kernel_size=25, strides=1, use_bias=False, use_bn=True, training=training
    )

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):

        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
            print(real_images)

        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)

                fake_logits = self.discriminator(fake_images, training=True)

                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)

            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class WGAN_train:
    def __init__(self, EPOCH, BATCH_SIZE):
        self.g_model = get_generator_model()
        self.d_model = get_discriminator_model()

        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE

    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def train(self, dataset, allpath):
        generator_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )

        discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )

        cbk = GANMonitor(allpath, num_img=3, latent_dim=noise_dim)

        wgan = WGAN(
            discriminator=self.d_model,
            generator=self.g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=3,
        )

        wgan.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
            g_loss_fn=self.generator_loss,
            d_loss_fn=self.discriminator_loss
        )

        wgan.fit(dataset, batch_size=self.BATCH_SIZE, epochs=self.EPOCH, callbacks=[cbk])

        os.makedirs('weights', exist_ok=True)
        self.g_model.save_weights('weights/generator_1.h5', True)
        self.d_model.save_weights('weights/discriminator_1.h5', True)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, allpath, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.allpath = glob.glob(allpath+'*')

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        img_list = []

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img_list.append(img)

        reals_p = np.random.choice(self.allpath, 3)
        reals = [np.load(reals_p[i]) for i in range(3)]

        os.makedirs('fig', exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(10, 15), constrained_layout=True)
        fig.suptitle('#{} Epoch'.format(epoch + 1), fontsize=20)

        for i in range(3):
            axes[i, 0].set_ylim(-0.75, 0.75)

            axes[i, 0].grid(True, which='major', color='lightgray', linestyle='dashed', linewidth=1, zorder=0)

            axes[i, 0].plot(img_list[i][:, 0], 'r')

            axes[i, 0].set_title('Generated')

            axes[i, 1].set_ylim(-0.75, 0.75)

            axes[i, 1].grid(True, which='major', color='lightgray', linestyle='dashed', linewidth=1, zorder=0)

            axes[i, 1].plot(reals[i][:, 0], 'b')

            axes[i, 1].set_title('Real')

        plt.savefig('fig/image_at_epoch_{:04d}.png'.format(epoch))


class AnoGAN:

    ### anomaly loss function
    def sum_of_residual(self, y_true, y_pred):
        return K.sum(K.abs(y_true - y_pred))

    ### discriminator intermediate layer feautre extraction
    def feature_extractor(self):
        d = get_discriminator_model(training=False)
        d.load_weights('weights/discriminator_1.h5')
        intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
        intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        return intermidiate_model

    ### anomaly detection model define
    def anomaly_detector(self):
        g = get_generator_model(training=False)
        g.load_weights('weights/generator_1.h5')
        intermidiate_model = self.feature_extractor()
        intermidiate_model.trainable = False
        g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
        g.trainable = False
        # Input layer can't be trained. Add new layer as same size & same distribution
        aInput = Input(shape=(32,))
        gInput = Dense((32), trainable=True)(aInput)
        gInput = Activation('sigmoid')(gInput)

        # G & D feature
        G_out = g(gInput)
        D_out = intermidiate_model(G_out)
        model = Model(inputs=aInput, outputs=[G_out, D_out])
        model.compile(loss=self.sum_of_residual, loss_weights=[0.90, 0.10], optimizer='rmsprop')

        return model

    ### anomaly detection
    def compute_anomaly_score(self, intermidiate_model, x, iterations=500, d=None):
        z = np.random.uniform(0, 1, size=(1, 32))
        d_x = intermidiate_model(x)

        model = self.anomaly_detector()
        loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
        similar_data, _ = model(z)
        loss = loss.history['loss'][-1]

        return loss, similar_data

    def compute_test_loss(self, test_path, intermidiate_model):
        allpath = glob.glob(test_path + '*')

        losses = []
        for i, p in enumerate(allpath):
            loss, similar_data = self.compute_anomaly_score(intermidiate_model, np.expand_dims(np.load(p), axis=0),
                                                     iterations=100)
            losses += [loss]
            print(i, loss)

        np.save('./test', np.array(losses))

        print(np.mean(losses))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--training", dest='training', default=False)
    parser.add_argument("--epoch", dest='EPOCH', type=int, default=50)
    parser.add_argument("--batch_size", dest='BATCH_SIZE', type=int, default=32)
    parser.add_argument("--raw_path", dest='raw_path', default='raw/')
    parser.add_argument("--train_path", dest='train_path', default='train_data/')
    parser.add_argument("--test_path", dest='test_path', default='test_data/')

    args = parser.parse_args()

    if args.training == True:
        dl = DataLoader(args.raw_path, args.train_path, args.test_path)
        dl.split_train_test()
        dataset = dl.make_dataset(args.train_path, args.BATCH_SIZE)
        wgan_train = WGAN_train(args.EPOCH, args.BATCH_SIZE)
        wgan_train.train(dataset, args.train_path)

    anoGAN = AnoGAN()
    intermidiate_model = anoGAN.feature_extractor()
    anoGAN.compute_test_loss(args.test_path, intermidiate_model)

