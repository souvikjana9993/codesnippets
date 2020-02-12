"""
mlxtend ExhaustiveFeatureSelector, selects best subset of features from all possible combinations of the features
link :http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/
"""
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
efs_1 = EFS(knn,min_features=1,max_features=4,scoring='accuracy',print_progress=True,cv=5) ##where knn is the model

efs_1.get_metric_dict()
print('Selected features:', efs1.best_idx_)

"""
Sequential feature selection is better computationally compared to EFS, however they should not be applied along with embedded feature selection methods like LASSO
Compared to RFE its more computation intensive as it relies on a metric for feature selection,
whereas RFE relies on weight coefficients(linear) or feature importances(tree based algos)

There are 4 different flavors of SFAs available via the SequentialFeatureSelector:

Sequential Forward Selection (SFS)
Sequential Backward Selection (SBS)
Sequential Forward Floating Selection (SFFS)
Sequential Backward Floating Selection (SBFS)
link: http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
"""
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(knn,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=0)

sfs1 = sfs1.fit(X, y)
sfs1.k_feature_idx_
sfs1.k_feature_names_
sfs1.k_score_
sfs.get_metric_dict()
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
