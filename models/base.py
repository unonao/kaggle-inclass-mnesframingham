from abc import abstractmethod
from typing import List


class Model:
    '''
    モデルのための抽象基底クラス
    '''
    @abstractmethod
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        raise NotImplementedError

    '''
    @abstractmethod
    def train_without_validation(self, train, weight, categorical_features: List[str], target: str, params: dict, best_iteration: int):
        raise NotImplementedError
    '''
