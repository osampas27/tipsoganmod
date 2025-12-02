import tensorflow as tf
from tensorflow.keras import layers
class FeatureMHSA(layers.Layer):
    def __init__(self, num_heads=4, key_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Dense(key_dim)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm = layers.LayerNormalization()
    def call(self, x, training=False):
        x = tf.expand_dims(x, axis=1)
        x = self.proj(x)
        attn = self.mha(x, x, x, training=training)
        out = self.norm(attn + x)
        out = tf.reduce_mean(out, axis=1)
        return out
