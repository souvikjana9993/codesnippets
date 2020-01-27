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
    
"""

ordinal encoding example 
X,Xt are train and target respectively
ordinal_vars are the column list of ordinal variables
ref : https://www.kaggle.com/lucamassaron/catboost-beats-auto-ml
"""

ordinals = {
    'ord_1' : {
        'Novice' : 0,
        'Contributor' : 1,
        'Expert' : 2,
        'Master' : 3,
        'Grandmaster' : 4
    },
    'ord_2' : {
        'Freezing' : 0,
        'Cold' : 1,
        'Warm' : 2,
        'Hot' : 3,
        'Boiling Hot' : 4,
        'Lava Hot' : 5
    }
}

def return_order(X, Xt, var_name):
    mode = X[var_name].mode()[0]
    el = sorted(set(X[var_name].fillna(mode).unique())|set(Xt[var_name].fillna(mode).unique()))
    return {v:e for e, v in enumerate(el)}

for mapped_var in ordinal_vars:
    if mapped_var not in ordinals:
        mapped_values = return_order(X, Xt, mapped_var)
        X[mapped_var + '_num'] = X[mapped_var].replace(mapped_values)
        Xt[mapped_var + '_num'] = Xt[mapped_var].replace(mapped_values)
    else:
        X[mapped_var + '_num'] = X[mapped_var].replace(ordinals[mapped_var])
        Xt[mapped_var + '_num'] = Xt[mapped_var].replace(ordinals[mapped_var])

"""
frequency encoding 
df is the train data
ref : https://www.kaggle.com/lucamassaron/catboost-beats-auto-ml
"""

# Enconding frequencies instead of labels (so we have some numeric variables)
def frequency_encoding(column, df, df_test=None):
    frequencies = df[column].value_counts().reset_index()
    df_values = df[[column]].merge(frequencies, how='left', 
                                   left_on=column, right_on='index').iloc[:,-1].values
    if df_test is not None:
        df_test_values = df_test[[column]].merge(frequencies, how='left', 
                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values
    else:
        df_test_values = None
    return df_values, df_test_values

for column in X.columns:
    train_values, test_values = frequency_encoding(column, X, Xt)
    X[column+'_counts'] = train_values
    Xt[column+'_counts'] = test_values
    
"""
target encoding example 
X,Xt are train and target respectively
ref :https://www.kaggle.com/lucamassaron/catboost-beats-auto-ml
"""

# Target encoding of selected variables
X['fold_column'] = 0
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137)

import category_encoders as cat_encs

cat_feat_to_encode = binary_vars + ordinal_vars + nominal_vars + time_vars
smoothing = 0.3

enc_x = np.zeros(X[cat_feat_to_encode].shape)

for i, (tr_idx, oof_idx) in enumerate(kf.split(X, y)):
    encoder = cat_encs.TargetEncoder(cols=cat_feat_to_encode, smoothing=smoothing)
    
    X.loc[oof_idx, 'fold_column'] = i
    
    encoder.fit(X[cat_feat_to_encode].iloc[tr_idx], y[tr_idx])
    enc_x[oof_idx, :] = encoder.transform(X[cat_feat_to_encode].iloc[oof_idx], y[oof_idx])
    
encoder.fit(X[cat_feat_to_encode], y)
enc_xt = encoder.transform(Xt[cat_feat_to_encode]).values

for idx, new_var in enumerate(cat_feat_to_encode):
    new_var = new_var + '_enc'
    X[new_var] = enc_x[:,idx]
    Xt[new_var] = enc_xt[:, idx]
