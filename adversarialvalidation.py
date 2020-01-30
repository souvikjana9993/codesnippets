#https://www.kaggle.com/tunguz/adversicat-ii



import xgboost as xgb
xtrain = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')
id_train = xtrain['Accident_ID']
ytrain = xtrain['Severity']
xtrain.drop(['Severity','Accident_ID'], axis = 1, inplace = True)
xtrain.fillna(-999, inplace = True)


xtest = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')
id_test = xtest['Accident_ID']
xtest.drop(['Accident_ID'], axis = 1, inplace = True)
xtest.fillna(-999, inplace = True)


# add identifier and combine
xtrain['istrain'] = 1
xtest['istrain'] = 0
xdat = pd.concat([xtrain, xtest], axis = 0)

# convert non-numerical columns to integers
df_numeric = xdat.select_dtypes(exclude=['object'])
df_obj = xdat.select_dtypes(include=['object']).copy()
    
for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]
    
xdat = pd.concat([df_numeric, df_obj], axis=1)
y = xdat['istrain']; xdat.drop('istrain', axis = 1, inplace = True)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)
xgb_params = {
        'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,
        'colsample_bytree': 0.9,'objective': 'binary:logistic',
        'silent': 1, 'n_estimators':100, 'gamma':1,
        'min_child_weight':4
        }   
clf = xgb.XGBClassifier(**xgb_params, seed = 10)



for train_index, test_index in skf.split(xdat, y):
        x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]
        y0, y1 = y.iloc[train_index], y.iloc[test_index]        
        print(x0.shape)
        clf.fit(x0, y0, eval_set=[(x1, y1)],
               eval_metric='logloss', verbose=False,early_stopping_rounds=10)
                
        prval = clf.predict(x1)
        print(roc_auc_score(y1,prval,average='weighted'))
