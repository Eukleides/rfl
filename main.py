import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
RANDOM_DIM = 500
DATADIR = "Data/PetImages"
CATEGORIES = ["Dog"]
IMG_SIZE = 150

def create_training_data():
    training_data = []

    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                # normalize our inputs to be in the range[-1, 1]
                new_array1 = (new_array.astype(np.float32) - 127.5) / 127.5
                new_array1 = new_array1.reshape(IMG_SIZE*IMG_SIZE)

                training_data.append(new_array1)
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    random.shuffle(training_data)

    ntest = int(0.9*len(training_data))
    x_train = training_data[0:ntest]
    x_test = training_data[ntest:]
    return (x_train, x_test)

# You will use the Adam optimizer
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=RANDOM_DIM, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(IMG_SIZE*IMG_SIZE, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=IMG_SIZE*IMG_SIZE, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, in_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(in_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=4, dim=(2, 2), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, RANDOM_DIM])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, IMG_SIZE, IMG_SIZE)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/gan_generated_image_epoch_%d.png' % epoch)
    plt.show()

def plot_generated_images2(examples=4, dim=(2, 2), figsize=(10, 10)):
    from keras.models import load_model
    generator = load_model('results\generator_model_215')

    noise = np.random.normal(0, 1, size=[examples, RANDOM_DIM])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, IMG_SIZE, IMG_SIZE)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_image(X, n):
    img = X[n].reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(img, cmap='gray')  # graph it

def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train, x_test = create_training_data()
    # Split the training data into batches of size 128
    batch_count = int(len(x_train) / batch_size)

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, RANDOM_DIM, generator, adam)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, RANDOM_DIM])
            image_batch = np.array(random.sample(x_train, batch_size))

            # Generate fake images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, RANDOM_DIM])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 5 == 0:
            #plot_generated_images(e, generator)
            generator.save('results/generator_model_%d' % e)

if __name__ == '__main__':
    #train(1000, 128)
    plot_generated_images2()
