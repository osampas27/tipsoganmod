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
        self.input_dim = input_dim
        self.latent_dim = 100

        # Build modules
        self.gen  = build_generator(self.latent_dim, input_dim)
        self.disc = build_discriminator(input_dim)
        self.dee  = build_deepred(
            input_dim=input_dim,
            num_heads=cfg.num_heads,
            key_dim=cfg.key_dim,
            use_attention=use_attention,
        )

        # Optimizers
        self.opt_g = optimizers.Adam(cfg.lr, beta_1=0.5, beta_2=0.999)
        self.opt_d = optimizers.Adam(cfg.lr, beta_1=0.5, beta_2=0.999)

        # Dynamic loss combiner
        self.loss_combiner = DynamicLossCombiner(cfg.alpha, cfg.beta, cfg.gamma)

        # History
        self.history = {"gen_loss": [], "disc_loss": []}

    def set_static_loss_weights(self, a=0.5, b=0.5, g=0.0):
        self.loss_combiner = DynamicLossCombiner(a, b, g)

    def _sample_latent(self, n):
        return tf.random.normal((n, self.latent_dim))

    # ============================================================
    #   SAFE PRETRAIN (ALSO FIXED)
    # ============================================================
    def pretrain_psogan(self, X_normal, epochs=3, batch_size=128):
        # If too few normals, tile to ensure enough samples
        if X_normal.shape[0] < batch_size:
            factor = int(np.ceil(batch_size / max(1, X_normal.shape[0])))
            X_normal = np.tile(X_normal, (factor, 1))

        dataset = (
            tf.data.Dataset.from_tensor_slices(X_normal.astype(np.float32))
            .shuffle(10000)
            .repeat()
            .batch(batch_size, drop_remainder=True)
        )
        it = iter(dataset)

        for _ in range(epochs):
            for _ in range(10):  # 10 batches enough for pretrain stability
                try:
                    real = next(it)
                except StopIteration:
                    real = X_normal[:batch_size]

                z = self._sample_latent(tf.shape(real)[0])

                with tf.GradientTape() as td, tf.GradientTape() as tg:
                    fake = self.gen(z, training=True)
                    d_real, _ = self.disc(real, training=True)
                    d_fake, _ = self.disc(fake, training=True)
                    d_loss, g_adv = adversarial_losses(d_real, d_fake, smoothing=True)
                    recon = tf.reduce_mean(tf.abs(real - fake))
                    g_loss = self.loss_combiner.combine(g_adv, recon, 0.0)

                self.opt_d.apply_gradients(
                    zip(td.gradient(d_loss, self.disc.trainable_variables),
                        self.disc.trainable_variables)
                )
                self.opt_g.apply_gradients(
                    zip(tg.gradient(g_loss, self.gen.trainable_variables),
                        self.gen.trainable_variables)
                )

    def pso_initialize(self, X_batch):
        # PSO init can stay empty or be implemented
        pass

    # ============================================================
    #   SAFE TRAIN TIPSO-GAN (MAIN FIX)
    # ============================================================
    def train_tipso(
        self,
        X_normal,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=5,
        batch_size=128,
        balance_strategy="none",
        collect_loss=True,
        use_pso=True,
    ):
        # Pretrain the deep classifier
        self.dee.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["acc"],
        )
        self.dee.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=batch_size,
            verbose=0,
        )

        # ==== SAFE NORMAL DATA HANDLING ====
        if X_normal.shape[0] < batch_size:
            factor = int(np.ceil(batch_size / max(1, X_normal.shape[0])))
            X_normal_expanded = np.tile(X_normal, (factor, 1))
        else:
            X_normal_expanded = X_normal

        # TensorFlow dataset
        dataset = (
            tf.data.Dataset.from_tensor_slices(X_normal_expanded.astype(np.float32))
            .shuffle(10000)
            .repeat()
            .batch(batch_size, drop_remainder=True)
        )
        it = iter(dataset)

        # Steps per epoch safe computation
        steps_per_epoch = max(1, X_normal_expanded.shape[0] // batch_size)

        # ---- PSO init ----
        if use_pso:
            try:
                first_batch = next(it)
            except StopIteration:
                first_batch = X_normal_expanded[:batch_size]
            self.pso_initialize(first_batch)

        # =======================================================
        #   MAIN TIPSO TRAIN LOOP
        # =======================================================
        best_acc = 0.0
        drops = 0

        for ep in range(epochs):
            ep_g = []
            ep_d = []

            for _ in range(steps_per_epoch):
                try:
                    real = next(it)
                except StopIteration:
                    real = X_normal_expanded[:batch_size]

                z = self._sample_latent(tf.shape(real)[0])

                with tf.GradientTape() as td, tf.GradientTape() as tg:
                    fake = self.gen(z, training=True)
                    d_real, f_real = self.disc(real, training=True)
                    d_fake, f_fake = self.disc(fake, training=True)

                    d_loss_adv, g_adv = adversarial_losses(
                        d_real, d_fake, smoothing=True
                    )

                    # focal loss (stabilizer)
                    y_real = tf.zeros_like(d_real)
                    y_fake = tf.ones_like(d_fake)
                    fl = focal_loss(
                        tf.concat([y_real, y_fake], axis=0),
                        tf.concat([d_real, d_fake], axis=0),
                        alpha=cfg.focal_alpha,
                        gamma=cfg.focal_gamma,
                    )
                    d_loss = d_loss_adv + fl

                    # reconstruction
                    recon = reconstruction_loss(real, f_real, self.gen)
                    g_loss = self.loss_combiner.combine(g_adv, recon, 0.0)

                # Apply gradients
                self.opt_d.apply_gradients(
                    zip(td.gradient(d_loss, self.disc.trainable_variables),
                        self.disc.trainable_variables)
                )
                self.opt_g.apply_gradients(
                    zip(tg.gradient(g_loss, self.gen.trainable_variables),
                        self.gen.trainable_variables)
                )

                if collect_loss:
                    ep_g.append(float(g_loss.numpy()))
                    ep_d.append(float(d_loss.numpy()))

            # Record epoch losses
            if collect_loss:
                self.history["gen_loss"].append(float(np.mean(ep_g)))
                self.history["disc_loss"].append(float(np.mean(ep_d)))

            # Validation accuracy gate
            preds = self.dee.predict(X_val, verbose=0)
            acc = np.mean(np.argmax(preds, axis=1) == y_val.reshape(-1))

            print(f"[TIPSO] epoch {ep+1}/{epochs} val-acc={acc:.4f} (best={best_acc:.4f})")

            if acc > best_acc:
                best_acc = acc
            else:
                drops += 1
                if drops >= cfg.patience_drop or best_acc >= cfg.val_acc_threshold:
                    print("[TIPSO] Early stop by validation gate.")
                    break
