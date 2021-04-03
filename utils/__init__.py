import pandas as pd
import time
import contextlib
from sklearn.metrics import accuracy_score, log_loss ,roc_auc_score

@contextlib.contextmanager
def simple_timer():
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_feather(f'data/interim/train.feather')
    y_train = train[target_name]
    return y_train

def evaluate_score(true, predicted, metric_name):
    if metric_name == 'acc':
        if predicted.ndim == 1:
            pred = predicted
        elif predicted.ndim == 2:
            pred = predicted.argmax(axis=1)
        return accuracy_score(true, pred)
    elif metric_name == 'logloss':
        return log_loss(true, predicted)
    elif metric_name == 'auc':
        return roc_auc_score(true, predicted)
