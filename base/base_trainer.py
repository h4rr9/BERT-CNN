class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model

        self.train_data = data.get_train_data()
        self.train_label = data.get_train_label()
        self.n_train = data.get_train_count()

        self.val_data = data.get_val_data()
        self.val_label = data.get_val_label()
        self.n_val = data.get_val_count()

        self.test_data = data.get_test_data()
        self.n_test = data.get_test_count()

        self.config = config

    def train(self):
        raise NotImplementedError
