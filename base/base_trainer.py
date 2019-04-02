class BaseTrain(object):
    def __init__(self, model, generator, config):
        self.model = model
        self.generator = generator
        self.config = config

    def train(self):
        raise NotImplementedError
