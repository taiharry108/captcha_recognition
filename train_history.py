import json
import pandas as pd


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
        self.reset()

    def reset(self):
        self.epoch_train_loss = 0
        self.epoch_train_correct = 0
        self.epoch_train_n = 0
        self.epoch_test_loss = 0
        self.epoch_test_correct = 0
        self.epoch_test_n = 0

    def accmulate_train_history(self, loss, correct, n):
        self.epoch_train_correct += correct
        self.epoch_train_loss += loss * n
        self.epoch_train_n += n

    def accmulate_test_history(self, loss, correct, n):
        self.epoch_test_correct += correct
        self.epoch_test_loss += loss * n
        self.epoch_test_n += n

    def _add_train_history(self, loss, acc):
        self.train_losses.append(loss)
        self.train_acc.append(acc)

    def _add_test_history(self, loss, acc):
        self.test_losses.append(loss)
        self.test_acc.append(acc)
    
    def epoch_finish(self):
        train_loss = self.epoch_train_loss / self.epoch_train_n
        train_acc = self.epoch_train_correct / self.epoch_train_n

        test_loss = self.epoch_test_loss / self.epoch_test_n
        test_acc = self.epoch_test_correct / self.epoch_test_n
        
        self._add_train_history(train_loss, train_acc)
        self._add_test_history(test_loss, test_acc)

        self.reset()


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
        return cls(model_name=history_dict['model'],
                   train_losses=history_dict['train_losses'],
                   test_losses=history_dict['test_losses'],
                   train_acc=history_dict['train_acc'],
                   test_acc=history_dict['test_acc'],
                   )
    
    def get_df(self):
        return pd.DataFrame(
            {
                "train loss": self.train_losses,
                "train acc": self.train_acc,
                "test loss": self.test_losses,
                "test acc": self.test_acc
            })
