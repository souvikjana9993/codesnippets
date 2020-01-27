"""
ref: https://www.kaggle.com/tunguz/cat-ii-histgradientboostingclassifier-baseline
Crossvalidation for numpy array type train,target
note train does not have target
"""
train_oof = np.zeros((train.shape[0],))
test_preds = 0

%%time
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137)

for jj, (train_index, val_index) in enumerate(kf.split(train)):
    print("Fitting fold", jj+1)
    train_features = train[train_index]
    train_target = target[train_index]
    
    val_features = train[val_index]
    val_target = target[val_index]
    
    model = HistGradientBoostingClassifier(max_iter=10000, learning_rate=0.01)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)
    train_oof[val_index] = val_pred[:,1]
    print("Fold AUC:", roc_auc_score(val_target, val_pred[:,1]))
    test_preds += model.predict_proba(test)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()
