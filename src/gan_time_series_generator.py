import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, LSTM, Conv1D, UpSampling1D, Activation
from tensorflow.keras.layers import MaxPooling1D, Bidirectional
import numpy as np
import matplotlib.pyplot as plt


class GANTimeSeriesGenerator:
    def __init__(self, input_dim=187, noise_dim=50, learning_rate=1e-4, 
                 generator_version='G_v1', discriminator_version='D_v1', 
                 noise_type='random'):
        self.input_dim = input_dim  # Shape of the time series data (e.g., 187 for ECG data)
        self.noise_dim = noise_dim  # Size of the random noise input to the generator
        self.learning_rate = learning_rate
        self.generator_version = generator_version
        self.discriminator_version = discriminator_version
        self.noise_type = noise_type  # 'random' or 'sinusoidal'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(self.discriminator_version)
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(self.generator_version)

        # Build the combined GAN model
        self.discriminator.trainable = False
        noise = Input(shape=(self.noise_dim,))
        generated_data = self.generator(noise)
        validity = self.discriminator(generated_data)
        self.gan = models.Model(noise, validity)
        self.gan.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                         loss='binary_crossentropy')

    def build_generator(self, version='G_v1'):
        if version == 'G_v1':
            model = models.Sequential(name='Generator_v1')
            model.add(Reshape((self.noise_dim, 1), input_shape=(self.noise_dim,)))
            model.add(Bidirectional(LSTM(16, return_sequences=True)))
            model.add(Conv1D(32, kernel_size=8, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(UpSampling1D())
            model.add(Conv1D(16, kernel_size=8, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(UpSampling1D())
            model.add(Conv1D(8, kernel_size=8, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(Conv1D(1, kernel_size=8, padding="same"))
            model.add(Flatten())

            model.add(Dense(self.input_dim))
            model.add(Activation('sigmoid'))
            model.add(Reshape((self.input_dim, 1)))

        model.summary()
        return model

    def build_discriminator(self, version='D_v1'):
        model = models.Sequential(name='Discriminator_v1')
        model.add(Conv1D(8, kernel_size=3, strides=1, input_shape=(self.input_dim, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Conv1D(32, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        return model

    def generate_noise(self, batch_size):
        if self.noise_type == 'sinusoidal':
            x = np.linspace(-np.pi, np.pi, self.noise_dim)
            noise = 0.1 * np.random.random_sample((batch_size, self.noise_dim)) + 0.9 * np.sin(x)
        else:  # Default to random noise
            noise = np.random.normal(0, 1, size=(batch_size, self.noise_dim))
        return noise

    def train(self, data, epochs=1000, batch_size=32, verbose=100):
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]
            real_data = np.expand_dims(real_data, axis=-1)  # Add channel dimension

            noise = self.generate_noise(batch_size)
            generated_data = self.generator.predict(noise, verbose=0)

            d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.gan.train_on_batch(noise, real_labels)

            if epoch % verbose == 0:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f}] [G loss: {g_loss:.4f}]")

    def generate_samples(self, n_samples=100):
        noise = self.generate_noise(n_samples)
        generated_data = self.generator.predict(noise, verbose=0)
        return generated_data

    def plot_samples(self, real_data, n_samples=5):
        generated_data = self.generate_samples(n_samples)
        
        plt.figure(figsize=(10, 6))
        for i in range(n_samples):
            plt.plot(real_data[i], label=f'Real {i}', alpha=0.7)
            plt.plot(generated_data[i], label=f'Synthetic {i}', linestyle='--', alpha=0.8)
        
        plt.title('Real vs. Synthetic Samples')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()




# import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, LSTM, Conv1D, UpSampling1D, Activation
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models
# from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Bidirectional, LSTM, Conv1D, UpSampling1D, Activation
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, Bidirectional, LSTM, Conv1D, UpSampling1D, Activation, MaxPooling1D
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models, Input
# from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv1D, MaxPooling1D, LSTM, Bidirectional, UpSampling1D, Activation
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv1D, MaxPooling1D, LSTM, Bidirectional, UpSampling1D, Activation

# class GANTimeSeriesGenerator:
#     def __init__(self, input_dim=187, noise_dim=50, learning_rate=1e-4, generator_version='G_v1', discriminator_version='D_v1'):
#         self.input_dim = input_dim  # Shape of the time series data (e.g., 187 for ECG data)
#         self.noise_dim = noise_dim  # Size of the random noise input to the generator
#         self.learning_rate = learning_rate
#         self.generator_version = generator_version
#         self.discriminator_version = discriminator_version

#         # Build and compile the discriminator
#         self.discriminator = self.build_discriminator(self.discriminator_version)
#         self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
#                                    loss='binary_crossentropy',
#                                    metrics=['accuracy'])

#         # Build the generator
#         self.generator = self.build_generator(self.generator_version)

#         # Build the combined GAN model
#         self.discriminator.trainable = False
#         noise = Input(shape=(self.noise_dim,))
#         generated_data = self.generator(noise)
#         validity = self.discriminator(generated_data)
#         self.gan = models.Model(noise, validity)
#         self.gan.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
#                          loss='binary_crossentropy')

#     def build_generator(self, version='G_v1'):
#         if version == 'G_v1':
#             model = models.Sequential(name='Generator_v1')
#             model.add(Reshape((self.noise_dim, 1), input_shape=(self.noise_dim,)))
#             model.add(Bidirectional(LSTM(16, return_sequences=True)))
#             model.add(Conv1D(32, kernel_size=8, padding="same"))
#             model.add(LeakyReLU(alpha=0.2))

#             model.add(UpSampling1D())
#             model.add(Conv1D(16, kernel_size=8, padding="same"))
#             model.add(LeakyReLU(alpha=0.2))

#             model.add(UpSampling1D())
#             model.add(Conv1D(8, kernel_size=8, padding="same"))
#             model.add(LeakyReLU(alpha=0.2))

#             model.add(Conv1D(1, kernel_size=8, padding="same"))
#             model.add(Flatten())

#             model.add(Dense(self.input_dim))
#             model.add(Activation('sigmoid'))
#             model.add(Reshape((self.input_dim, 1)))

#         elif version == 'G_v2':
#             model = models.Sequential(name='Generator_v2')
#             model.add(Reshape((self.noise_dim, 1)))
#             model.add(Bidirectional(LSTM(1, return_sequences=True)))
#             model.add(Flatten())
#             model.add(Dense(100))
#             model.add(LeakyReLU(alpha=0.2))
#             model.add(Dense(150))
#             model.add(LeakyReLU(alpha=0.2))
#             model.add(Dense(self.input_dim))
#             model.add(Activation('sigmoid'))
#             model.add(Reshape((self.input_dim, 1)))

#         elif version == 'G_v3':
#             model = models.Sequential(name='Generator_v3')
#             model.add(Reshape((self.noise_dim, 1)))
#             model.add(LSTM(50, return_sequences=True))
#             model.add(LSTM(50, return_sequences=True))
#             model.add(Flatten())
#             model.add(Dense(self.input_dim))
#             model.add(Activation('sigmoid'))
#             model.add(Reshape((self.input_dim, 1)))

#         model.summary()
#         return model

#     def build_discriminator(self, version='D_v1'):
#             if version == 'D_v1':
#                 model = models.Sequential(name='Discriminator_v1')
#                 model.add(Conv1D(8, kernel_size=3, strides=1, input_shape=(self.input_dim, 1), padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3))

#                 model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3, strides=2))

#                 model.add(Conv1D(32, kernel_size=3, strides=2, padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3, strides=2))

#                 model.add(Flatten())
#                 model.add(Dense(1, activation='sigmoid'))

#             elif version == 'D_v2':
#                 model = models.Sequential(name='Discriminator_v2')
#                 model.add(Conv1D(3, kernel_size=3, strides=1, input_shape=(self.input_dim, 1), padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3))

#                 model.add(Conv1D(5, kernel_size=3, strides=1, padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3, strides=2))

#                 model.add(Conv1D(8, kernel_size=3, strides=2, padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3, strides=2))

#                 model.add(Conv1D(12, kernel_size=3, strides=2, padding='same'))
#                 model.add(LeakyReLU(alpha=0.2))
#                 model.add(MaxPooling1D(3, strides=2))

#                 model.add(Flatten())
#                 model.add(Dense(1, activation='sigmoid'))

#             elif version == 'D_v3':
#                 model = models.Sequential(name='Discriminator_v3')
#                 model.add(Conv1D(filters=32, kernel_size=16, strides=1, padding='same'))
#                 model.add(LeakyReLU())
#                 model.add(Conv1D(filters=64, kernel_size=16, strides=1, padding='same'))
#                 model.add(LeakyReLU())
#                 model.add(MaxPooling1D(pool_size=2))
#                 model.add(Conv1D(filters=128, kernel_size=16, strides=1, padding='same'))
#                 model.add(LeakyReLU())
#                 model.add(Conv1D(filters=256, kernel_size=16, strides=1, padding='same'))
#                 model.add(LeakyReLU())
#                 model.add(MaxPooling1D(pool_size=2))
#                 model.add(Flatten())
#                 model.add(Dense(1, activation='sigmoid'))

#             model.summary()
#             return model

#     def train(self, data, epochs=1000, batch_size=32, verbose=100):
#         real_labels = np.ones((batch_size, 1))
#         fake_labels = np.zeros((batch_size, 1))

#         for epoch in range(epochs):
#             idx = np.random.randint(0, data.shape[0], batch_size)
#             real_data = data[idx]
#             real_data = np.expand_dims(real_data, axis=-1)  # Add channel dimension

#             noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
#             generated_data = self.generator.predict(noise, verbose=0)

#             d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
#             d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

#             g_loss = self.gan.train_on_batch(noise, real_labels)

#             if epoch % verbose == 0:
#                 print(f"{epoch} [D loss: {(d_loss_real[0] + d_loss_fake[0]) / 2:.4f}, acc: {(d_loss_real[1] + d_loss_fake[1]) / 2:.4f}] [G loss: {g_loss:.4f}]")

#     def generate_samples(self, n_samples=100):
#         noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
#         generated_data = self.generator.predict(noise, verbose=0)
#         return generated_data

#     def plot_samples(self, real_data, n_samples=5):
#         generated_data = self.generate_samples(n_samples)
        
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 6))
#         for i in range(n_samples):
#             plt.plot(real_data[i], label=f'Real {i}', alpha=0.7)
#             plt.plot(generated_data[i], label=f'Synthetic {i}', linestyle='--', alpha=0.8)
        
#         plt.title('Real vs. Synthetic Samples')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Amplitude')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


