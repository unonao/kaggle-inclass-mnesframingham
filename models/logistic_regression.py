from sklearn.linear_model import LogisticRegression
import logging

import pandas as pd
from models.base import Model
from sklearn.metrics import roc_auc_score

class LogisticRegressionClassifier(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):

        # ロガーの作成
        logger = logging.getLogger('main')

        model = LogisticRegression(penalty=params["penalty"],C=params["C"], max_iter=params["max_iter"], random_state=params["seed"])
        model.fit(X_train, y_train)

        # valid を予測する
        y_valid_pred = model.predict_proba(X_valid)
        # テストデータを予測する
        y_pred = model.predict_proba(X_test)

        return y_pred, y_valid_pred, model
