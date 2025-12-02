import tensorflow as tf
from tensorflow.keras import layers, models
def build_generator(latent_dim=100, output_dim=120):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(output_dim, activation='tanh')(x)
    return models.Model(inputs, x, name='GeneratorTabular')
def build_discriminator(input_dim=120):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    feat = layers.Dense(64, activation='relu', name='features')(x)
    logits = layers.Dense(1)(feat)
    return models.Model(inputs, [logits, feat], name='DiscriminatorTabular')
