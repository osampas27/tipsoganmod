import tensorflow as tf
from tensorflow.keras import layers
def focal_loss(labels, logits, alpha=1.0, gamma=2.0):
    y = tf.cast(labels, tf.float32)
    p = tf.sigmoid(logits)
    pt = tf.where(tf.equal(y,1.0), p, 1.0 - p)
    w = alpha * tf.pow(1.0 - pt, gamma)
    return tf.reduce_mean(-w * tf.math.log(pt + 1e-7))
def adversarial_losses(d_logits_real, d_logits_fake, smoothing=True):
    if smoothing:
        real_labels = tf.ones_like(d_logits_real) * 0.9
        fake_labels = tf.zeros_like(d_logits_fake) + 0.1
    else:
        real_labels = tf.ones_like(d_logits_real)
        fake_labels = tf.zeros_like(d_logits_fake)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=d_logits_real))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=d_logits_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_logits_fake), logits=d_logits_fake))
    return d_loss, g_loss
def reconstruction_loss(real_x, feat, generator):
    latent = layers.Dense(100)(feat)
    x_hat = generator(latent, training=True)
    return tf.reduce_mean(tf.abs(real_x - x_hat))
class DynamicLossCombiner:
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.0):
        s = alpha + beta + gamma if (alpha + beta + gamma) != 0 else 1.0
        self.alpha = alpha/s
        self.beta = beta/s
        self.gamma = gamma/s
    def combine(self, adv, recon, stability=0.0):
        return self.alpha*adv + self.beta*recon + self.gamma*stability
