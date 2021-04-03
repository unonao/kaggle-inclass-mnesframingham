import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

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


class Original(Feature):
    def create_features(self):
        self.train = train
        self.test = test

class FixingSkewnessOriginal(Feature):
    def create_features(self):
        df = features.copy()
        df = fixing_skewness(df)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]

if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    train = train.drop(["index","TenYearCHD"],axis=1) # index,target を落とす
    test = pd.read_feather('./data/interim/test.feather')
    test = test.drop(["index"],axis=1) # index を落とす
    features = pd.concat([train, test])

    print(train.head())

    generate_features(globals(), args.force)
