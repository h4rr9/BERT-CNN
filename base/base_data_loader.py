class BaseDataLoader(object):
    def __init__(self, config):
        self.confog = config

    def get_train_data(self):
        raise NotImplementedError

    def get_val_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

