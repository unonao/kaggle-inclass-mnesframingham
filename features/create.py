import numpy as np
import pandas as pd
import re as re

from base import Feature, get_arguments, generate_features
from target_encoder import KFoldTargetEncoderTrain, TargetEncoderTest

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
Feature.dir = 'features'

"""
# original features
male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose
"""

def fixing_skewness(df):
    """
    This function takes in a dataframe and return fixed skewed dataframe
    """
    # Getting all the data that are not of "object" type.
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
    return df

def fill_features_na(df):
    """
    education          105
    cigsPerDay          29
    BPMeds              53
    totChol             50
    BMI                 19
    heartRate            1
    glucose            388
    """
    return df.fillna(df.median)

class Original(Feature):
    def create_features(self):
        self.train = train
        self.test = test

# target encode
class currentSmoker_target(Feature): # 効果無し
    def create_features(self):
        feature_name = "currentSmoker"
        target_encode_name = f'{feature_name}_Kfold_Target_Enc'
        targetc = KFoldTargetEncoderTrain(feature_name,'TenYearCHD',n_fold=5)
        new_train = targetc.fit_transform(train_with_target)
        test_targetc = TargetEncoderTest(new_train, feature_name, target_encode_name)
        new_test = test_targetc.fit_transform(test)
        self.train[target_encode_name] = new_train[target_encode_name]
        self.test[target_encode_name] = new_test[target_encode_name]
class diabetes_target(Feature): # 効果無し
    def create_features(self):
        feature_name = "diabetes"
        target_encode_name = f'{feature_name}_Kfold_Target_Enc'
        targetc = KFoldTargetEncoderTrain(feature_name,'TenYearCHD',n_fold=5)
        new_train = targetc.fit_transform(train_with_target)
        test_targetc = TargetEncoderTest(new_train, feature_name, target_encode_name)
        new_test = test_targetc.fit_transform(test)
        self.train[target_encode_name] = new_train[target_encode_name]
        self.test[target_encode_name] = new_test[target_encode_name]
class prevalentHyp_target(Feature): # 効果あり！！
    def create_features(self):
        feature_name = "prevalentHyp"
        target_encode_name = f'{feature_name}_Kfold_Target_Enc'
        targetc = KFoldTargetEncoderTrain(feature_name,'TenYearCHD',n_fold=5)
        new_train = targetc.fit_transform(train_with_target)
        test_targetc = TargetEncoderTest(new_train, feature_name, target_encode_name)
        new_test = test_targetc.fit_transform(test)
        self.train[target_encode_name] = new_train[target_encode_name]
        self.test[target_encode_name] = new_test[target_encode_name]
class prevalentStroke_target(Feature):  # 効果無し
    def create_features(self):
        feature_name = "prevalentStroke"
        target_encode_name = f'{feature_name}_Kfold_Target_Enc'
        targetc = KFoldTargetEncoderTrain(feature_name,'TenYearCHD',n_fold=5)
        new_train = targetc.fit_transform(train_with_target)
        test_targetc = TargetEncoderTest(new_train, feature_name, target_encode_name)
        new_test = test_targetc.fit_transform(test)
        self.train[target_encode_name] = new_train[target_encode_name]
        self.test[target_encode_name] = new_test[target_encode_name]
class education_target(Feature): # 効果無し
    def create_features(self):
        feature_name = "education"
        target_encode_name = f'{feature_name}_Kfold_Target_Enc'
        targetc = KFoldTargetEncoderTrain(feature_name,'TenYearCHD',n_fold=5)
        new_train = targetc.fit_transform(train_with_target)
        test_targetc = TargetEncoderTest(new_train, feature_name, target_encode_name)
        new_test = test_targetc.fit_transform(test)
        self.train[target_encode_name] = new_train[target_encode_name]
        self.test[target_encode_name] = new_test[target_encode_name]
class bpmeds_target(Feature): # 効果無し
    def create_features(self):
        feature_name = "BPMeds"
        target_encode_name = f'{feature_name}_Kfold_Target_Enc'
        targetc = KFoldTargetEncoderTrain(feature_name,'TenYearCHD',n_fold=5)
        new_train = targetc.fit_transform(train_with_target)
        test_targetc = TargetEncoderTest(new_train, feature_name, target_encode_name)
        new_test = test_targetc.fit_transform(test)
        self.train[target_encode_name] = new_train[target_encode_name]
        self.test[target_encode_name] = new_test[target_encode_name]



# poly から考えた特徴量
class AgeSysbp(Feature):
    def create_features(self):
        self.train["age_sysBP"] = train["age"]*train["sysBP"]
        self.test["age_sysBP"] = test["age"]*test["sysBP"]
class Female08Sysbp(Feature):
    def create_features(self):
        female = features["male"].copy()
        female[female==0] = 0.8
        df = female * features["sysBP"]
        self.train["female08_sysBP"] = df[:train.shape[0]]
        self.test["female08_sysBP"] = df[train.shape[0]:]
class Age2Diabp(Feature):
    def create_features(self):
        self.train["age2_diaBP"] = train["age"]**2 *train["diaBP"]
        self.test["age2_diaBP"] = test["age"]**2 *test["diaBP"]
class Female085Age(Feature):
    def create_features(self):
        female = features["male"].copy()
        female[female==0] = 0.85
        df = female * features["age"]
        self.train["female085_age"] = df[:train.shape[0]]
        self.test["female085_age"] = df[train.shape[0]:]
class Noprevalenthyp085Age(Feature):
    def create_features(self):
        noprevalentHyp = features["prevalentHyp"].copy()
        noprevalentHyp[noprevalentHyp==0] = 0.85
        df = noprevalentHyp * features["age"]
        self.train["noprevalenthyp085_age"] = df[:train.shape[0]]
        self.test["noprevalenthyp085_age"] = df[train.shape[0]:]
class Female08Diabp(Feature):
    def create_features(self):
        female = features["male"].copy()
        female[female==0] = 0.8
        df = female * features["diaBP"]
        self.train["female08_diaBP"] = df[:train.shape[0]]
        self.test["female08_diaBP"] = df[train.shape[0]:]
class Nosmoker092Diabp(Feature):
    def create_features(self):
        nosmoker = features["currentSmoker"].copy()
        nosmoker[nosmoker==0] = 0.92
        df = nosmoker * features["diaBP"]
        self.train["nosmoker092_diaBP"] = df[:train.shape[0]]
        self.test["nosmoker092_diaBP"] = df[train.shape[0]:]
class Noprevalenthyp085Diabp(Feature):
    def create_features(self):
        noprevalentHyp = features["prevalentHyp"].copy()
        noprevalentHyp[noprevalentHyp==0] = 0.85
        df = noprevalentHyp * features["diaBP"]
        self.train["noprevalenthyp085_diaBP"] = df[:train.shape[0]]
        self.test["noprevalenthyp085_diaBP"] = df[train.shape[0]:]
class Nodiabetes065Diabp(Feature):
    def create_features(self):
        nodiabetes = features["diabetes"].copy()
        nodiabetes[nodiabetes==0] = 0.85
        df = nodiabetes * features["diaBP"]
        self.train["nodiabetes065_diaBP"] = df[:train.shape[0]]
        self.test["nodiabetes065_diaBP"] = df[train.shape[0]:]
class Nodiabetes065Sysbp(Feature):
    def create_features(self):
        nodiabetes = features["diabetes"].copy()
        nodiabetes[nodiabetes==0] = 0.85
        df = nodiabetes * features["sysBP"]
        self.train["nodiabetes065_sysBP"] = df[:train.shape[0]]
        self.test["nodiabetes065_sysBP"] = df[train.shape[0]:]


class FixingSkewnessOriginal(Feature): # 効果無し
    def create_features(self):
        df = features.copy()
        df = fixing_skewness(df)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]

class Pca(Feature):    # 効果あり
    def create_features(self):
        df = fill_features_na(features.copy()) # 欠損値を穴埋め
        scaler = StandardScaler()
        # scaler = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_feats]), columns=numeric_feats)
        n_pca = 2
        pca_cols = ["pca"+str(i) for i in range(n_pca)]
        pca = PCA(n_components=n_pca)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=pca_cols)
        pca_df[pca_cols] = scaler.fit_transform(pca_df[pca_cols])
        self.train = pca_df[:train.shape[0]].reset_index(drop=True)
        self.test = pca_df[train.shape[0]:].reset_index(drop=True)
        '''
        n_tsne = 2
        tsne_cols = ["tsne"+str(i) for i in range(n_tsne)]
        embeded = pd.DataFrame(bhtsne.tsne(all_df[numeric_features].astype(np.float64), dimensions=n_tsne, rand_seed=10), columns=tsne_cols)
        features = pd.concat([scaled_df, pca_df, embeded], axis=1)
        '''


class Polynomial2d(Feature): # うまくいった。もう少し絞るのが良さそう
    def create_features(self):
        df = fill_features_na(features.copy()) # 欠損値を穴埋め
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        df = df[numeric_feats]
        poly = PolynomialFeatures(2)
        original_fea_num = df.shape[1]
        poly_np = poly.fit_transform(df)[:, original_fea_num+1:]
        poly_features = poly.get_feature_names(df.columns)[original_fea_num+1:]
        poly_df = pd.DataFrame(poly_np, columns=poly_features)
        # fixed skew
        poly_df = fixing_skewness(poly_df)
        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)

class Polynomial3d(Feature):
    def create_features(self):
        df = fill_features_na(features.copy()) # 欠損値を穴埋め
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        df = df[numeric_feats]
        poly = PolynomialFeatures(3)
        original_fea_num = df.shape[1]
        poly_np = poly.fit_transform(df)[:, original_fea_num+1:]
        poly_features = poly.get_feature_names(df.columns)[original_fea_num+1:]
        poly_df = pd.DataFrame(poly_np, columns=poly_features)
        # fixed skew
        poly_df = fixing_skewness(poly_df)
        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)

class FraminghamRiskScore(Feature): # 効果なし
    # https://www.nhlbi.nih.gov/sites/default/files/media/docs/risk-assessment.pdf
    """
    HDL は無し
    """
    def create_features(self):
        df = pd.DataFrame()
        # women
        women_col_coef = {
            "LnAge":-29.799,
            "LnTotalCho":13.540,
            "LnAge_LnTotalCho":-13.578,
            "LnUntreatedSBP":1.957,
            "CurrentSmoker":7.574,
            "LnAge_CurrentSmoker":-1.665,
            "Diabetes":0.661,
            "BaselineSurvival":0.9665
        }
        men_col_coef = {
            "LnAge":12.344,
            "LnTotalCho":11.853 ,
            "LnAge_LnTotalCho":-2.664,
            "LnUntreatedSBP":1.764,
            "CurrentSmoker":7.837,
            "LnAge_CurrentSmoker":-1.795 ,
            "Diabetes":0.658 ,
            "BaselineSurvival":0.9144
        }
        df["LnAge"] = np.log(features["age"])
        df["LnTotalCho"] = np.log(features["totChol"])
        df["LnAge_LnTotalCho"] = df["LnAge"] * df["LnTotalCho"]
        df["LnUntreatedSBP"] = np.log(features["sysBP"])
        df["CurrentSmoker"] = features["currentSmoker"]
        df["LnAge_CurrentSmoker"] = df["LnAge"] * df["CurrentSmoker"]
        df["Diabetes"] = features["diabetes"]
        df["Sum"] = 0
        for is_male in [0,1]:
            col_coef = women_col_coef if is_male==0 else men_col_coef
            for key, coer in col_coef.items():
                if key != "BaselineSurvival":
                    df.loc[features["male"]==is_male, key] = coer * df.loc[features["male"]==is_male, key]
                    df.loc[features["male"]==is_male, "Sum"] += df.loc[features["male"]==is_male, key]
                else:
                    df.loc[features["male"]==is_male, "Sum"] -= df.loc[features["male"]==is_male, "Sum"].mean()
                    df.loc[features["male"]==is_male, "Ind"] = np.power(coer,np.exp(df.loc[features["male"]==is_male, "Sum"]))

        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]

if __name__ == '__main__':
    args = get_arguments()
    train_with_target = pd.read_feather('./data/interim/train.feather')
    train = train_with_target.drop(["index","TenYearCHD"],axis=1) # index,target を落とす
    test = pd.read_feather('./data/interim/test.feather')
    test = test.drop(["index"],axis=1) # index を落とす
    features = pd.concat([train, test])

    print(features.head())
    print(features.info())

    generate_features(globals(), args.force)
