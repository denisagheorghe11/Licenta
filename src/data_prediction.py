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

        

