from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from trainers.learning_rate_scd import step_decay_wrapper



class CosLossModelTrainer(BaseTrain):
    def __init__(self, model, train_generator, valid_generator, config):
        super(CosLossModelTrainer, self).__init__(model, train_generator, config)
        self.valid_generator = valid_generator
        self.callbacks = []
        self.loss = []
        self.val_loss = []
        self.step_decay_function = step_decay_wrapper(self.config)
        self.init_callbacks()


    def init_callbacks(self):
        #filepath = os.path.join(self.config.callbacks.checkpoint_dir,
        #                        '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name)
        if self.config.callbacks.is_save_model:
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.config.callbacks.checkpoint_dir, self.config.exp.name + '.hdf5'),
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

        self.callbacks.append(
            EarlyStopping(
                monitor='val_loss', min_delta=0.1, patience=4, verbose=1, mode='auto')
        )

        if self.config.trainer.learning_rate_schedule_type == 'ReduceLROnPlateau':
            self.callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=self.config.trainer.lr_decrease_factor,
                                            patience=2, min_lr=1e-09)
            )
        elif self.config.trainer.learning_rate_schedule_type == 'LearningRateScheduler':
            self.callbacks.append(
                LearningRateScheduler(self.step_decay_function)
            )



        #if hasattr(self.config,"comet_api_key"):
        #    from comet_ml import Experiment
        #    experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
        #    experiment.disable_mp()
        #    experiment.log_multiple_params(self.config)
        #    self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        history = self.model.fit_generator(
            self.generator,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=len(self.generator),
            validation_data=self.valid_generator,
            validation_steps=len(self.valid_generator),
            callbacks=self.callbacks,
            verbose=1)

        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])
