from base.base_trainer import BaseTrain
import os
import tensorflow as tf


class CoLAModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(CoLAModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.matt_corr = []
        self.val_loss = []
        self.val_acc = []
        self.val_matt_corr = []
        self.init_callbacks()
        self.initialize_vars(self.sess)

    def initialize_vars(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        tf.keras.backend.set_session(sess)

    def init_callbacks(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):

        self.initialize_vars(self.sess)

        history = self.model.fit(
            x=self.train_data,
            steps_per_epoch=self.n_train // self.config.data_loader.batch_size + 1,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.val_data,
            validation_steps=self.n_val // self.config.data_loader.batch_size + 1,
            callbacks=self.callbacks
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.matt_corr.extend(history.history['matt_corr'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        self.val_matt_corr.extend(history.history['val_matt_corr'])
