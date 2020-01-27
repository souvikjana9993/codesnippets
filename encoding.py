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

"""
frequency encoding 
df is the train data
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
