import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from .config import TIPSOConfig
from .gan import build_generator, build_discriminator
from .deepred import build_deepred
from .losses import focal_loss, reconstruction_loss, adversarial_losses, DynamicLossCombiner

cfg = TIPSOConfig()

class TIPSOTrainer:
    def __init__(self, input_dim=120, use_attention=True):
        self.input_dim=input_dim
        self.latent_dim=100
        self.gen=build_generator(self.latent_dim,input_dim)
        self.disc=build_discriminator(input_dim)
        self.dee=build_deepred(input_dim=input_dim, num_heads=cfg.num_heads, key_dim=cfg.key_dim, use_attention=use_attention)
        self.opt_g=optimizers.Adam(cfg.lr, beta_1=0.5, beta_2=0.999)
        self.opt_d=optimizers.Adam(cfg.lr, beta_1=0.5, beta_2=0.999)
        self.loss_combiner=DynamicLossCombiner(cfg.alpha,cfg.beta,cfg.gamma)
        self.history={'gen_loss':[],'disc_loss':[]}

    def set_static_loss_weights(self,a=0.5,b=0.5,g=0.0):
        self.loss_combiner=DynamicLossCombiner(a,b,g)

    def _sample_latent(self,n):
        return tf.random.normal((n,self.latent_dim))

    def pretrain_psogan(self,X_normal,epochs=3,batch_size=128):
        dataset=tf.data.Dataset.from_tensor_slices(X_normal).shuffle(10000).batch(batch_size)
        for _ in range(epochs):
            for real in dataset:
                z=self._sample_latent(tf.shape(real)[0])
                with tf.GradientTape() as td, tf.GradientTape() as tg:
                    fake=self.gen(z,training=True)
                    d_logits_real,_=self.disc(real,training=True)
                    d_logits_fake,_=self.disc(fake,training=True)
                    d_loss,g_adv=adversarial_losses(d_logits_real,d_logits_fake,True)
                    recon=tf.reduce_mean(tf.abs(real-fake))
                    g_loss=self.loss_combiner.combine(g_adv,recon,0.0)
                self.opt_d.apply_gradients(zip(td.gradient(d_loss,self.disc.trainable_variables), self.disc.trainable_variables))
                self.opt_g.apply_gradients(zip(tg.gradient(g_loss,self.gen.trainable_variables), self.gen.trainable_variables))

    def pso_initialize(self, X_batch):
        pass

    def train_tipso(self, X_normal, X_train, y_train, X_val, y_val,
                    epochs=5, batch_size=128, balance_strategy='none', collect_loss=True, use_pso=True):
        self.dee.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        self.dee.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=batch_size, verbose=0)
        dataset = tf.data.Dataset.from_tensor_slices(X_normal).shuffle(10000).repeat().batch(batch_size)
        steps_per_epoch = max(1, int(max(1, X_normal.shape[0]) // batch_size))
        it = iter(dataset)
        if use_pso:
            real = next(it)
            self.pso_initialize(real)
        best = 0.0; drops = 0
        for ep in range(epochs):
            ep_g, ep_d = [], []
            for _ in range(steps_per_epoch):
                real = next(it)
                z = self._sample_latent(tf.shape(real)[0])
                with tf.GradientTape() as td, tf.GradientTape() as tg:
                    fake = self.gen(z, training=True)
                    d_logits_real, feat_real = self.disc(real, training=True)
                    d_logits_fake, feat_fake = self.disc(fake, training=True)
                    d_loss_adv, g_adv = adversarial_losses(d_logits_real, d_logits_fake, smoothing=True)
                    y_real = tf.zeros_like(d_logits_real); y_fake = tf.ones_like(d_logits_fake)
                    fl = focal_loss(tf.concat([y_real,y_fake],0),
                                    tf.concat([d_logits_real,d_logits_fake],0),
                                    alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
                    d_loss = d_loss_adv + fl
                    recon = reconstruction_loss(real, feat_real, self.gen)
                    g_loss = self.loss_combiner.combine(g_adv, recon, 0.0)
                self.opt_d.apply_gradients(zip(td.gradient(d_loss, self.disc.trainable_variables), self.disc.trainable_variables))
                self.opt_g.apply_gradients(zip(tg.gradient(g_loss, self.gen.trainable_variables), self.gen.trainable_variables))
                if collect_loss:
                    ep_g.append(float(g_loss.numpy())); ep_d.append(float(d_loss.numpy()))
            if collect_loss:
                self.history['gen_loss'].append(float(np.mean(ep_g)) if len(ep_g) else 0.0)
                self.history['disc_loss'].append(float(np.mean(ep_d)) if len(ep_d) else 0.0)
            preds = self.dee.predict(X_val, verbose=0)
            acc = np.mean(np.argmax(preds, axis=1).reshape(-1,1) == y_val.reshape(-1,1))
            print(f"[TIPSO] epoch {ep+1}/{epochs} val-acc={acc:.4f} (best={best:.4f})")
            if acc > best: best = acc
            else:
                drops += 1
                if drops >= cfg.patience_drop or best >= cfg.val_acc_threshold:
                    print("[TIPSO] Early stop by validation gate.")
                    break
