"""
simple label encoder tqdm
"""

%%time

X_train = train.copy()
X_test = test.copy()

for cc in tqdm_notebook(columns):
    le = LabelEncoder()
    le.fit(list(train[cc].values)+list(test[cc].values))
    X_train[cc] = le.transform(train[cc].values)
    X_test[cc] = le.transform(test[cc].values)
