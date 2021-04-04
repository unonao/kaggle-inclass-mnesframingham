import os
import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
import argparse
import json
import numpy as np

from utils import load_datasets, load_target, evaluate_score
from models import LightGBM
import matplotlib.pyplot as plt
import seaborn as sns

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

config_filename = os.path.splitext(os.path.basename(options.config))[0]

feats = config['features']
target_name = config['target_name']
model_name = config['model']
model_params = config['params']

# log の設定
now = datetime.datetime.now()

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("main")    #logger名mainを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/{config_filename}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)



def train_and_predict(X_train_all, y_train_all, X_test):

    oof_df = pd.DataFrame(index=[i for i in range(X_train_all.shape[0])], columns=[i for i in range(model_params["num_class"])])  # meta model の X_train に
    y_preds = []

    models = []
    auc_scores =[]
    acc_scores = []
    logloss_scores  = []

    kf = KFold(n_splits=5)
    for train_index, valid_index in kf.split(X_train_all):
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # train & inference
        if model_name=="lightgbm":
            classifier = LightGBM()
        else:
            logger.debug("No such model name")
            raise Exception
        y_pred, y_valid_pred, model = classifier.train_and_predict(X_train, X_valid, y_train, y_valid, X_test, model_params)


        # 結果の保存
        y_preds.append(y_pred)
        oof_df.iloc[valid_index,:] = y_valid_pred
        models.append(model)

        # スコア
        auc_valid = evaluate_score(y_valid, y_valid_pred[:,1], "auc")
        acc_valid = evaluate_score(y_valid, y_valid_pred.argmax(axis=1), "acc")
        logloss_valid = evaluate_score(y_valid, y_valid_pred[:,1], "logloss")
        logger.debug(f"\t auc:{auc_valid}, acc: {acc_valid}, logloss: {logloss_valid}")
        auc_scores.append(auc_valid)
        acc_scores.append(acc_valid)
        logloss_scores.append(logloss_valid)


    # lightgbmなら重要度の出力
    if model_name=="lightgbm":
        feature_imp_np = np.zeros(X_train_all.shape[1])
        for model in models:
            feature_imp_np += model.feature_importance()/len(models)
        feature_imp = pd.DataFrame(sorted(zip(feature_imp_np, X_train_all.columns)), columns=['Value', 'Feature'])
        #print(feature_imp)
        logger.debug(feature_imp)
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(f'./logs/plots/features_{config_filename}.png')

    # oof
    oof_df.to_csv(
        f'./data/output/oof_{config_filename}.csv',
        index=False
    )
    # CVスコア
    auc_score = sum(auc_scores) / len(auc_scores)
    acc_score = sum(acc_scores) / len(acc_scores)
    logloss_score = sum(logloss_scores) / len(logloss_scores)
    logger.debug('===CV scores===')
    logger.debug(f"\t auc:{auc_score}, acc: {acc_score}, logloss: {logloss_score}")


    # submitファイルの作成
    ID_name = config['ID_name']
    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])
    y_sub = sum(y_preds) / len(y_preds)
    sub[target_name] = y_sub[:, 1]
    ''' 確率ではなく番号を出力
    if y_sub.shape[1] > 1:
        y_sub = np.argmax(y_sub, axis=1)
    '''
    sub = sub.rename(columns={ID_name: 'Id', target_name:"label"})
    sub.to_csv(
        f'./data/output/sub_{config_filename}.csv',
        index=False
    )

def main():

    logger.debug('config: {}'.format(options.config))
    logger.debug(feats)
    logger.debug(model_params)

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(feats)
    y_train_all = load_target(target_name)
    logger.debug("X_train_all shape: {}".format(X_train_all.shape))
    train_and_predict(X_train_all, y_train_all, X_test)


if __name__ == '__main__':
    main()
