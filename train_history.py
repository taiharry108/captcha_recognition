import json


class TrainHistory(object):
    def __init__(self, model_name, train_losses=None,
                 test_losses=None,
                 train_acc=None,
                 test_acc=None):
        self.model_name = model_name
        self.train_losses = [] if train_losses is None else train_losses
        self.test_losses = [] if test_losses is None else test_losses
        self.train_acc = [] if train_acc is None else train_acc
        self.test_acc = [] if test_acc is None else test_acc

    def add_train_history(self, loss, acc):
        self.train_losses.append(loss)
        self.train_acc.append(acc)

    def add_test_history(self, loss, acc):
        self.test_losses.append(loss)
        self.test_acc.append(acc)

    def save_history(self):
        history = {
            "model": self.model_name,
            "train_losses": self.train_losses,
            "train_acc": self.train_acc,
            "test_losses": self.test_losses,
            "test_acc": self.test_acc,
        }
        with open('{}.json'.format(self.model_name), 'w') as f:
            json.dump(history, f)

    @classmethod
    def load_history(cls, filename):
        with open(filename) as f:
            history_dict = json.load(f)
        return cls(model_name=history['model'],
                   train_losses=history['train_losses'],
                   test_losses=history['test_losses'],
                   train_acc=history['train_acc'],
                   test_acc=history['test_acc'],
                   )
