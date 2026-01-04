# Implement generator and discriminator networks

import tensorflow as tf

# Here we won't be reusing layers
def generator(Z, hid_size, reuse=False):
    with tf.variable_scope("GAN/generator", reuse=reuse):
        # create fully connected hidden layers and output layer
        h1 = tf.layers.dense(Z, hid_size[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hid_size[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 2)
    
    return out

def discriminator(X, hid_size, reuse=False):
    with tf.variable_scope("GAN/discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hid_size[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hid_size[1], activation=tf.nn.leaky_relu)
        # inclusion of h3 has something to do with being able to visualize the transformed feature space in a 2D plane idrk
        h3 = tf.layers.dense(h2, 2)
        out = tf.layers.dense(h3, 1)
    
    return out