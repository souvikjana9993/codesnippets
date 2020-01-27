"""
fill missing tqdm notebook
"""

from tqdm import tqdm_notebook

columns = train.columns

for cc in tqdm_notebook(columns):
    train[cc] = train[cc].fillna(train[cc].mode()[0])
    test[cc] = test[cc].fillna(test[cc].mode()[0])
