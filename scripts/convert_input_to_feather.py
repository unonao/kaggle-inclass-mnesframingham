"""
    Firstly, we should convert train&test data.
"""
import pandas as pd

target = {
    'train': 'train',
    'test': 'test',
}

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for k, v in target.items():
    (pd.read_csv('./data/input/' + k + '.' + extension, encoding="utf-8")).reset_index()\
        .to_feather('./data/interim/' + v + '.feather')
