


def step_decay_wrapper(config):
    def step_decay(epoch):
        if epoch < config.trainer.learning_rate_epoch_change[0]:
            lrate = config.trainer.learning_rate_values[0]
        elif epoch < config.trainer.learning_rate_epoch_change[1]:
            lrate = config.trainer.learning_rate_values[1]
        else:
            lrate = config.trainer.learning_rate_values[2]
        return lrate
    return step_decay

