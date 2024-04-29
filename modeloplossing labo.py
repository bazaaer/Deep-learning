import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
import os
import time



# Laden en voorbewerken van de dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normaliseren naar het bereik [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)

# GAN parameters
latent_dim = 100
img_shape = (28, 28, 1)

# Bouwen van de generator
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(256),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(512),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(np.prod(img_shape), activation='tanh'),
    Reshape(img_shape)
])

# Bouwen van de discriminator
discriminator = Sequential([
    Flatten(input_shape=img_shape),
    Dense(512),
    LeakyReLU(alpha=0.2),
    Dense(256),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')
])

# Compileren van de discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Bouwen en compileren van het GAN-model
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
fake_img = generator(gan_input)
gan_output = discriminator(fake_img)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy')
discriminator.trainable = True

# Functie voor het opslaan van gegenereerde afbeeldingen
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5  # Unnormaliseren naar het bereik [0, 1]

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f'gan_images/mnist_{epoch}.png')
    plt.close()

# Training van het GAN-model
epochs = 10000
batch_size = 128
sample_interval = 1000
start_time = time.time()
for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress
    if epoch % sample_interval == 0:
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}, Time Elapsed: {elapsed_time:.2f} seconds')
        save_imgs(epoch)
    
    # Stop als de maximale tijd van 1 minuut is bereikt
    if elapsed_time >= 60:
        break