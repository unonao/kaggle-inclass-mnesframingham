import os
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import datetime
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,QuantileTransformer

import argparse
import json
import numpy as np

from utils import load_datasets, load_target, evaluate_score
from models import LightGBM, NeuralNet,LogisticRegressionClassifier, CNN1d
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

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
seed = config['params']["seed"]
ID_name = config['ID_name']

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



def train_and_predict(X_train_all, y_train_all, X_test, seed_num):
    model_params["seed"] = seed + seed_num
    oof_df = pd.DataFrame(index=[i for i in range(X_train_all.shape[0])], columns=[i for i in range(model_params["num_class"])])
    y_preds = []

    models = []
    auc_scores =[]
    acc_scores = []
    logloss_scores  = []

    kf = StratifiedKFold(n_splits=config["fold"], shuffle=True, random_state=model_params["seed"])
    for fold_num, (train_index, valid_index) in enumerate(kf.split(X_train_all,y_train_all)):
        logger.debug(f"FOLD: {fold_num}")
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # train & inference
        if model_name=="lightgbm":
            classifier = LightGBM()
        elif model_name == "nn":
            classifier = NeuralNet(seed_num, fold_num)
        elif model_name == "cnn1d":
            classifier = CNN1d(seed_num, fold_num)
        elif model_name == "logistic_regression":
            classifier = LogisticRegressionClassifier()
        else:
            logger.debug("No such model name")
            raise Exception

        if "sampling" in config:
            if config["sampling"]=="SMOTE":
                X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            elif config["sampling"]=="ADASYN":
                X_train, y_train = ADASYN().fit_resample(X_train, y_train)
            elif config["sampling"]=="RandomOverSampler":
                X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)
            else:
                raise

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

    # CVスコア
    auc_score = sum(auc_scores) / len(auc_scores)
    acc_score = sum(acc_scores) / len(acc_scores)
    logloss_score = sum(logloss_scores) / len(logloss_scores)
    logger.debug('=== CV scores ===')
    logger.debug(f"\t auc:{auc_score}, acc: {acc_score}, logloss: {logloss_score}")


    # submitファイルの作成
    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])
    y_sub = sum(y_preds) / len(y_preds)
    sub[target_name] = y_sub[:, 1]
    ''' 確率ではなく番号を出力
    if y_sub.shape[1] > 1:
        y_sub = np.argmax(y_sub, axis=1)
    '''

    return oof_df, sub

# stacking 用
def stack_load_df(stacking_name):
    oof_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i, path in enumerate(stacking_name):
        path = os.path.join("data/output/", "oof_"+path+".csv")
        one_df = pd.read_csv(path)
        one_df.columns=["0", f"oof{i}"]
        one_df = one_df[f"oof{i}"]
        oof_df = pd.concat([oof_df, one_df], axis=1)
    for i, path in enumerate(stacking_name):
        path = os.path.join("data/output/", "sub_"+path+".csv")
        one_df = pd.read_csv(path)
        one_df.columns=["id",f"test{i}"]
        one_df = one_df[f"test{i}"]
        test_df = pd.concat([test_df, one_df], axis=1)
    return oof_df, test_df

def main():

    logger.debug('config: {}'.format(options.config))
    logger.debug(feats)
    logger.debug(model_params)
    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(feats)
    y_train_all = load_target(target_name)
    cols = X_train_all.columns

    # stacking
    if "stacking" in config and config["stacking"]==True:
        oof_df, test_df = stack_load_df(config["stacking_name"])
        X_train_all = pd.concat([X_train_all,oof_df],axis=1)
        X_test = pd.concat([X_test,test_df],axis=1)

    if (model_name != "lightgbm") or ("sampling" in config):
        logger.debug("rank gauss")
        scaler = QuantileTransformer(n_quantiles=100, random_state=model_params["seed"],output_distribution="normal")
        all_df = pd.concat([X_train_all,X_test])
        all_df = all_df.fillna(all_df.median()) # 欠損値埋め
        all_df[cols] = scaler.fit_transform(all_df[cols]) # scale
        X_train_all = all_df[:X_train_all.shape[0]].reset_index(drop=True)
        X_test = all_df[X_train_all.shape[0]:].reset_index(drop=True)

    logger.debug("X_train_all shape: {}".format(X_train_all.shape))
    print(X_train_all.info())

    # seed ごとにループ
    class_cols = [i for i in range(model_params["num_class"])]
    oof_df = pd.DataFrame(index=[i for i in range(X_train_all.shape[0])], columns=class_cols)
    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])
    oof_df[class_cols] = 0
    sub[target_name] = 0
    for seed_num in range(config["seed_num"]):
        logger.debug(f"SEED: {seed_num}")
        one_oof_df, one_sub = train_and_predict(X_train_all, y_train_all, X_test, seed_num=seed_num)
        oof_df[class_cols] += one_oof_df[class_cols]/config["seed_num"]
        sub[target_name]+= one_sub[target_name]/config["seed_num"]

    auc_score = evaluate_score(y_train_all.values, oof_df.values[:,1], "auc")
    acc_score = evaluate_score(y_train_all.values, oof_df.values.argmax(axis=1), "acc")
    logloss_score = evaluate_score(y_train_all.values, oof_df.values[:,1], "logloss")
    logger.debug('=== OOF CV scores ===')
    logger.debug(f"\t auc:{auc_score}, acc: {acc_score}, logloss: {logloss_score}")

    sub = sub.rename(columns={ID_name: 'Id', target_name:"label"})
    oof_df.to_csv(
        f'./data/output/oof_{config_filename}.csv',
        index=False
    )
    sub.to_csv(
        f'./data/output/sub_{config_filename}.csv',
        index=False
    )

if __name__ == "__main__":
    main()
