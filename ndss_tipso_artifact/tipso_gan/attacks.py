# tipso_gan/attacks.py
import numpy as np
import tensorflow as tf

def _ensure_arrays(xmin, xmax):
    xmin = np.asarray(xmin, dtype=np.float32)
    xmax = np.asarray(xmax, dtype=np.float32)
    return xmin, xmax

def feature_bounds_from_data(X):
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    # avoid zero width
    width = np.maximum(xmax - xmin, 1e-8)
    return xmin.astype(np.float32), xmax.astype(np.float32), width.astype(np.float32)

def _clip_per_feature(x_adv, xmin, xmax):
    return tf.clip_by_value(x_adv, xmin, xmax)

def fgsm(model, x, y, eps, xmin, xmax, loss_fn=None):
    """
    Untargeted FGSM: x_adv = x + eps * sign(grad_x loss(model(x), y))
    x, y: numpy arrays (float32, int)
    eps: float or 1D array broadcastable to x
    xmin/xmax: per-feature bounds (float32 np arrays)
    """
    if loss_fn is None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.int32)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        logits = model(x_tf, training=False)
        loss = loss_fn(y_tf, logits)
    grad = tape.gradient(loss, x_tf)
    x_adv = x_tf + tf.sign(grad) * tf.cast(eps, tf.float32)
    x_adv = _clip_per_feature(x_adv, xmin, xmax)
    return x_adv.numpy()

def bim(model, x, y, eps, alpha, iters, xmin, xmax, loss_fn=None):
    """
    Basic Iterative Method (Projected FGSM).
    """
    if loss_fn is None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    x0 = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.int32)
    x_adv = tf.identity(x0)

    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv, training=False)
            loss = loss_fn(y_tf, logits)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + tf.sign(grad) * tf.cast(alpha, tf.float32)

        # project to L_inf ball around x0 with radius eps, then clip to feature bounds
        x_adv = tf.minimum(tf.maximum(x_adv, x0 - eps), x0 + eps)
        x_adv = _clip_per_feature(x_adv, xmin, xmax)

    return x_adv.numpy()

def pgd_linf(model, x, y, eps, alpha, iters, xmin, xmax, random_start=True, loss_fn=None):
    """
    PGD with L_inf constraint and projection, untargeted.
    """
    if loss_fn is None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    x0 = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.int32)

    if random_start:
        noise = tf.random.uniform(tf.shape(x0), minval=-eps, maxval=eps, dtype=tf.float32)
        x_adv = x0 + noise
    else:
        x_adv = tf.identity(x0)

    x_adv = _clip_per_feature(x_adv, xmin, xmax)

    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv, training=False)
            loss = loss_fn(y_tf, logits)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + tf.sign(grad) * tf.cast(alpha, tf.float32)

        # Project back to L_inf ball and clip
        x_adv = tf.minimum(tf.maximum(x_adv, x0 - eps), x0 + eps)
        x_adv = _clip_per_feature(x_adv, xmin, xmax)

    return x_adv.numpy()
