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
