_DATA_PATH = "./data/glue_data/"
# _DATA_PATH = "/scratch/scratch1/harig/data/glue_data/"


class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_data(self):
        raise NotImplementedError

    def get_val_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

    def get_train_count(self):
        raise NotImplementedError

    def get_val_count(self):
        raise NotImplementedError

    def get_test_count(self):
        raise NotImplementedError
