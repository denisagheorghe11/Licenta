import pands as pd
######################################################################
#
#
# Mihai - 2019
######################################################################
# load datasets
df = pd.read_sql_table()

######################################################################
# data transformation
#Now that the dataset is already in a pands dataframe, next we have to
df['target'] = dr['y'].apply(lambda x: 1 if x == 'yes' else 0)

######################################################################
# descriptive stats
# Exploratory statistics help a modeler understand the data better. A
#couple of these stats are available in this framework.

df.isnull().mean().sort_values(ascending=False)*100

# check the corellation between variables
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,
            yticklabels=corr.columns)

bar_color = '#058caa'
num_color = '#ed8549'

final_iv,_ = data_vars(df1,df1['target'])
final_iv = final_iv[(final_iv.VAR_NAME != 'target')]
grouped = final_iv.groupby(['VAR_NAME'])
for key, group in grouped:
    ax = group.plot('MIN_VALUE','EVENT_RATE',kind='bar',color=bar_color,linewidth=1.0,edgecolor=['black'])
    ax.set_title(str(key) + " vs " + str('target'))
    ax.set_xlabel(key)
    ax.set_ylabel(str('target') + " %")
    rects = ax.patches
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.01*height, str(round(height*100,1)) + '%',
                ha='center', va='bottom', color=num_color, fontweight='bold')

# Prediction Model
from sklearn.cross_validation import train_test_split

train, test = train_test_split(df1, test_size = 0.4)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

features_train = train[list(vif['Features'])]
label_train = train['target']
features_test = test[list(vif['Features'])]
label_test = test['target']

# Apply different algorithms on the train dataset and evaluate the performance on the test data
#to make sure the model is stable. The framework includes codes for RandomForest, Logistic Regression
#Naive Bayes, NeuralNetwork and Gradient Boosting. It can be applied other models based on our needs
#My choose it was RandomForest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

# Hyper parameter Tunning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(features_train, label_train)

# Final Model and Model performance
# The final model that gives us the better accuracy values is picked for now
pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])

#ROC or AUC curve or c-statistics
from bokeh.charts import Histogram
from ipywidgets import interact
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook
output_notebook()
from sklearn import metrics
preds = clf.predict_proba(features_train)[:,1]
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), preds)
auc = metrics.auc(fpr,tpr)
p = figure(title="ROC Curve - Train data")
r = p.line(fpr,tpr,color='#0077bc',legend = 'AUC = '+ str(round(auc,3)), line_width=2)
s = p.line([0,1],[0,1], color= '#d15555',line_dash='dotdash',line_width=2)
show(p)

#Decide Plots and Kolmogorov Smirnov (KS) statistics
deciling(scores_train,['DECILE'],'TARGET','NONTARGET')
#lift charts, actual vs prediction chart and gains chart
gains(lift_train,['DECILE'],'TARGET','SCORE')

#Save the model for future development of the code
import pandas
from sklearn.externals import joblib

filename = 'final_model.model'
i = [d,clf]
joblib.dump(i,filename)
#clf - is the model classifier object
#d   - is the label encoder object used to transform character to numeric values
