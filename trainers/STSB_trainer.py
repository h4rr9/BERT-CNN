from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf


class STSBModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(STSBModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.pear_corr = []
        self.spear_corr = []
        self.val_loss = []
        self.val_pear_corr = []
        self.val_spear_corr = []
        self.init_callbacks()

    def initialize_vars(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        K.set_session(sess)

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
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
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):

        history = self.model.fit(
            x=[self.train_data[0], self.train_data[0]],
            y=self.train_label,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=(
                [self.val_data[0], self.val_data[1]], self.val_label),
            callbacks=self.callbacks,
            batch_size=self.config.trainer.batch_size
        )

        self.loss.extend(history.history['loss'])
        self.pear_corr.extend(history.history['pear_corr'])
        # self.spear_corr.extend(history.history['spear_corr'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_pear_corr.extend(history.history['val_pear_corr'])
        # self.val_spear_corr.extend(history.history['val_spear_corr'])
