
#importing plotly
import chart_studio
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()
"""
plotly with cufflinks groupby barplot with buttons
"""
safetysummary=train_df.groupby('Severity')[['Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Violations','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection']].mean()


lists=[]
for idx,i in enumerate(safetysummary.columns):
    bool_array=[False]*len(safetysummary.columns)
    bool_array[idx]=True
    lists.append(
        dict(label=str(i),
             method="update",
             args=[{"visible":bool_array},
                   {"title":i}]))

layout=dict(
    updatemenus=list([
        dict(
            active=0,
            buttons=lists,
        )
    ])
)

safetysummary.iplot(kind='bar', xTitle='Severity', yTitle='Magnitude',title='Severity to mean scores',layout=layout)

"""
plot removing outliers
"""

def plot_after_outlier_removal(train_df):
    train_df2=train_df.copy()
    percentiles1 = train_df2['Control_Metric'].quantile([0.01,0.99]).values
    train_df2['Control_Metric'] = np.clip(train_df2['Control_Metric'], percentiles1[0], percentiles1[1])
    percentiles2 = train_df2['Turbulence_In_gforces'].quantile([0.01,0.99]).values
    train_df2['Turbulence_In_gforces'] = np.clip(train_df2['Turbulence_In_gforces'], percentiles2[0], percentiles2[1])
    train_df2[['Turbulence_In_gforces','Control_Metric']].iplot(kind='scatter',mode='markers', x='Control_Metric', y='Turbulence_In_gforces',title='Turbulence to Control')
"""
numeric boxplot
"""
box_age = train_df[['Safety_Score', 'Severity']]
box_age.pivot(columns='Severity', values='Safety_Score').iplot(kind='box')
"""
histogram numeric 
"""
train_data=train_df[['Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection']]

lists=[]
for idx,i in enumerate(train_data.columns):
    bool_array=[False]*len(train_data.columns)
    bool_array[idx]=True
    lists.append(
        dict(label=str(i)+" histogram",
             method="update",
             args=[{"visible":bool_array},
                   {"title":i}]))

layout=dict(
    updatemenus=list([
        dict(
            active=0,
            buttons=lists,
        )
    ])
)

#X_train[['Safety_Score','Control_Metric','Adverse_Weather_Metric']].iplot(kind="hist",title="ass",layout=layout)
train_df[['Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection']].iplot(kind="hist",title="ass",layout=layout)
