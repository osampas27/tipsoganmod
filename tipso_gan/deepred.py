import tensorflow as tf
from tensorflow.keras import layers, models
from .attention import FeatureMHSA
def build_deepred(input_dim=120, num_classes=2, num_heads=4, key_dim=16, use_attention=True):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    if use_attention:
        attn = FeatureMHSA(num_heads=num_heads, key_dim=key_dim)(x)
        fused = layers.Concatenate()([x, attn])
    else:
        fused = x
    fused = layers.Dense(128, activation='relu')(fused)
    outputs = layers.Dense(num_classes, activation='softmax')(fused)
    return models.Model(inputs, outputs, name='DeePredTabular')
