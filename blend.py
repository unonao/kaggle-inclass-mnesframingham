import numpy as np
import pandas as pd
ID_name = "Id"
target_name = "label"

sub = pd.DataFrame(pd.read_csv(f'data/input/submit_sample.csv')[ID_name])
sub[target_name] = 0

base_subs = {
    "data/output/sub_best_lightgbm.csv":1.,
    "data/output/sub_best_nn3.csv":1.,
    "data/output/sub_best_nn4.csv":2.,
    "data/output/sub_best_logreg_000.csv":1.,
}

sum_weight = 0
for path,w in base_subs.items():
    tmp_sub = pd.read_csv(path)
    sub[target_name] += tmp_sub[target_name] * w
    sum_weight += w

sub[target_name] /= sum_weight


sub.to_csv(
    './data/output/sub_blend.csv',
    index=False
)
