'''
同じように stratifyKFold で分割
configs を元に、all, clean, misc のモデルを推論
Lightgbm などを用いてアンサンブル
'''
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  log_loss

ID_name = "index"
target_name = "TenYearCHD"

# LightGBM parameters
params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'auc_mu'},
        'num_class': 2,
        'learning_rate': 0.01,
        'max_depth': 4,
        'num_leaves':3,
        'lambda_l2' : 0.3,
        'num_iteration': 2000,
        "min_data_in_leaf":1,
        'verbose': 0
}

CFG = {
    "fold_num": 5,
    "seed": 719,
}

# data path
oof_path = [
    "data/output/oof_best_lightgbm.csv",
    "data/output/oof_best_logreg_000.csv",
    "data/output/oof_best_nn3.csv",
    "data/output/oof_best_nn4.csv",
]
test_path = [
    "data/output/sub_best_lightgbm.csv",
    "data/output/sub_best_logreg_000.csv",
    "data/output/sub_best_nn3.csv",
    "data/output/sub_best_nn4.csv",
]

class LightGBM():
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        # データセットを生成する
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        model = lgb.train(
            params, lgb_train,
            # モデルの評価用データを渡す
            valid_sets=lgb_eval,
            # 50 ラウンド経過しても性能が向上しないときは学習を打ち切る
            early_stopping_rounds=50,
        )

        # valid を予測する
        y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        # テストデータを予測する
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        return y_pred, y_valid_pred, model

def load_df(oof_path, test_path):
    oof_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i, path in enumerate(oof_path):
        one_df = pd.read_csv(path)
        one_df.columns=["0", f"oof{i}"]
        one_df = one_df[f"oof{i}"]
        oof_df = pd.concat([oof_df, one_df], axis=1)
    for i, path in enumerate(test_path):
        one_df = pd.read_csv(path)
        one_df.columns=["id",f"test{i}"]
        one_df = one_df[f"test{i}"]
        test_df = pd.concat([test_df, one_df], axis=1)
    return oof_df, test_df

def load_train_label(path="data/input/train.csv"):
    train = pd.read_csv(path)
    return train[target_name]

def main():
    oof_df, test_df = load_df(oof_path, test_path)
    oof_label = load_train_label()

    y_preds = []
    scores_loss = []
    scores_acc = []
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(oof_df.shape[0]), oof_label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_train, X_valid = (oof_df.iloc[trn_idx, :], oof_df.iloc[val_idx, :])
        y_train, y_valid = (oof_label.iloc[trn_idx], oof_label.iloc[val_idx])

        model = LightGBM()

        # 学習と推論
        y_pred, y_valid_pred, m = model.train_and_predict(X_train, X_valid, y_train, y_valid, test_df, params)

        # 結果の保存
        y_preds.append(y_pred)

        # スコア
        loss = log_loss(y_valid, y_valid_pred)
        scores_loss.append(loss)
        acc = (y_valid == np.argmax(y_valid_pred, axis=1)).mean()
        scores_acc.append(acc)
        print(f"\t log loss: {loss}")
        print(f"\t acc: {acc}")

    loss = sum(scores_loss) / len(scores_loss)
    print('===CV scores loss===')
    print(scores_loss)
    print(loss)
    acc = sum(scores_acc) / len(scores_acc)
    print('===CV scores acc===')
    print(scores_acc)
    print(acc)

    tst_preds = np.mean(y_preds, axis=0)

    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])
    y_sub = sum(tst_preds) / len(tst_preds)
    sub[target_name] = y_sub[:, 1]
    sub = sub.rename(columns={ID_name: 'Id', target_name:"label"})


    sub.to_csv(f'output/submission_ensemble.csv', index=False)

if __name__ == '__main__':
    main()
