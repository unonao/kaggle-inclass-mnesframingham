import numpy as np
import pandas as pd
ID_name = "Id"
target_name = "label"

sub = pd.DataFrame(pd.read_csv(f'data/input/submit_sample.csv')[ID_name])
sub[target_name] = 0

base_subs = {
    "lightgbm": "data/output/sub_best001.csv",
    "nn": "data/output/sub_best_nn000.csv",
    "logreg":  "data/output/sub_best_logreg_000.csv",
}

for base, path in base_subs.items():
    tmp_sub = pd.read_csv(path)
    sub[target_name] += tmp_sub[target_name]
sub[target_name] /= len(base_subs)


sub.to_csv(
    './data/output/sub_blend.csv',
    index=False
)
