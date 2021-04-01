#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#To remove warning
import warnings
warnings.filterwarnings('ignore')


# In[28]:


#Load the dataset
data=pd.read_excel("sample_service.xlsx")


# In[29]:


#The method .copy() is used here so that any changes made in new DataFrame don't get reflected in the original one
inc=data.copy()


# In[30]:


#To head the first 5 rows
inc.head()


# In[31]:


#To read the last 5 rows
inc.tail()


# In[32]:


inc.describe()


# In[33]:


inc.info()


# ### Pre-Processing the dataset

# In[34]:


#Replace ? with the nan
inc=inc.replace("?",np.nan)


# In[35]:


inc.isna().sum(axis=0)


# In[36]:


#Removing extra string
inc["ID_caller"]= inc["ID_caller"].str.replace("Caller", " ") 
inc["opened_by"]= inc["opened_by"].str.replace("Opened by", " ") 
inc["Created_by"]= inc["Created_by"].str.replace("Created by", " ") 
inc["Category Id"]= inc["Category Id"].str.replace("Subcategory", " ") 
inc["user_symptom"]=inc["user_symptom"].str.replace("Symptom", " ") 
inc["Support_group"]=inc["Support_group"].str.replace("Group", " ") 
inc["support_incharge"]=inc["support_incharge"].str.replace("Resolver", " ") 
inc["problem_ID"]=inc["problem_ID"].str.replace("Problem ID", " ") 
inc["updated_by"]= inc["updated_by"].str.replace("Updated by", " ") 


# In[37]:


#Rename the column name
inc.rename({'Category Id':'Category_id'},axis=1, inplace=True)


# In[79]:


inc["target_impact"]=inc["impact"].apply(lambda x: int(x.split(' ')[0]))


# In[83]:


inc["location"]= inc["location"].str.replace("Location", " ") 


# ### Visualisation of Dataset

# In[54]:


#visulaisation for categorical data


# In[38]:


import seaborn as sns
sns.countplot(inc['impact'])


# In[39]:


# Visulaization for categorical attributes
plt.figure(figsize=(15,8))

sns.countplot(inc['ID_status'])


# In[40]:


sns.countplot(inc['active'])


# In[41]:


sns.countplot(inc['Doc_knowledge'])


# In[42]:


sns.countplot(inc['confirmation_check'])


# In[43]:


sns.countplot(inc['notify'])


# In[49]:


sns.countplot(inc['type_contact'])


# In[53]:


plt.figure(figsize=(25,8))
sns.countplot(inc['location'])


# In[44]:


# Visulaization for numerical attributes


# In[45]:


def dist(inc,var1):
    plt.figure()
    sns.distplot(inc[var1],kde = False,bins = 30)
    plt.show()


# In[46]:


dist(inc,"count_reassign")


# In[47]:


dist(inc, "count_opening")
open_val_count = inc['count_opening'].value_counts()
print(open_val_count)
print(open_val_count[0]/len(inc))


# In[48]:


dist(inc, "count_updated")
updated_val_count = inc['count_updated'].value_counts()
print(updated_val_count[0:20])
print("Most updated count :", updated_val_count.index.max())
print(updated_val_count[0:10].sum()/len(inc))


# In[55]:


#check for the missing values


# In[57]:


cols =inc.columns 
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(inc[cols].isnull(),
            cmap=sns.color_palette(colours))


# In[70]:


# Value Counts


# In[71]:


inc['active'].value_counts()


# In[72]:


inc['count_reassign'].value_counts()


# In[73]:


inc['count_reassign'].value_counts()


# In[74]:


inc['count_opening'].value_counts()


# In[75]:


inc['count_updated'].value_counts()


# In[76]:


inc['Doc_knowledge'].value_counts()


# In[77]:


inc['confirmation_check'].value_counts()


# In[78]:


inc['notify'].value_counts()


# In[82]:


#format the date & columns
inc["updated_day"]=pd.to_datetime(inc.updated_at).dt.day
inc["updated_month"]=pd.to_datetime(inc.updated_at).dt.month
inc["updated_year"]=pd.to_datetime(inc.updated_at).dt.year
inc["updated_hr"]=pd.to_datetime(inc.updated_at).dt.hour
inc["updated_minute"]=pd.to_datetime(inc.updated_at).dt.minute
inc["opened_at_day"]=pd.to_datetime(inc.opened_time).dt.day
inc["opened_at_month"]=pd.to_datetime(inc.opened_time).dt.month
inc["opened_at_year"]=pd.to_datetime(inc.opened_time).dt.year
inc["opened_at_hr"]=pd.to_datetime(inc.opened_time).dt.hour
inc["opened_at_minute"]=pd.to_datetime(inc.opened_time).dt.minute
inc["created_at_day"]=pd.to_datetime(inc.created_at).dt.day
inc["created_at_month"]=pd.to_datetime(inc.created_at).dt.month
inc["created_at_year"]=pd.to_datetime(inc.created_at).dt.year
inc["created_at_hr"]=pd.to_datetime(inc.created_at).dt.hour
inc["created_at_minute"]=pd.to_datetime(inc.created_at).dt.minute


# In[91]:


for col_name in inc.columns: 
    print ("column:",col_name,".Missing:",sum(inc[col_name].isnull()))


# In[113]:


inc2=inc.copy()


# In[114]:


#Drop the columns which are not required as opened_time,created_at,updated_at we have split it datetime format we dont need these columns and problem_id,change_request_support_incharge have more than 50% missing hence we will remove this column too
inc2.drop(['opened_time','created_at','updated_at','support_incharge','change_request','problem_ID'],axis=1,inplace=True)


# ### Label Encoding

# In[115]:


from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
inc2['ID_status']=number.fit_transform(inc2['ID_status'])
inc2['ID_status'].value_counts()


# In[116]:


inc2['active']=number.fit_transform(inc2['active'])
inc2['active'].value_counts()


# In[117]:


inc2['type_contact']=number.fit_transform(inc2['type_contact'])
inc2['type_contact'].value_counts()


# In[118]:


inc2['Doc_knowledge']=number.fit_transform(inc2['Doc_knowledge'])
inc2['Doc_knowledge'].value_counts()


# In[119]:


inc2['confirmation_check']=number.fit_transform(inc1['confirmation_check'])
inc2['confirmation_check'].value_counts()


# In[134]:


#Filling na values with the median values
for columns in ['user_symptom','created_at_day','created_at_month','created_at_year','created_at_hr','created_at_minute']:
    median=inc2[columns].median()
    inc2[columns]=inc2[columns].fillna(median)


# In[122]:


inc2.drop(['impact'],axis=1,inplace=True)


# In[126]:


inc2.drop(['notify'],axis=1,inplace=True)


# In[127]:


inc2['ID']=inc2['ID'].str.replace("INC", " ") 
inc2.head(2)


# In[136]:


inc2['Created_by'] = inc2['Created_by'].astype(float)
inc2['Created_by'].fillna((inc2['Created_by'].mean()), inplace=True)
inc2['Created_by'] = inc2['Created_by'].astype(int)


# In[139]:


inc2['ID']=inc2['ID'].astype(int)
inc2['location']=inc2['location'].astype(int)
inc2['ID']=inc2['ID'].astype(float).astype(int)
inc2['ID_caller']=inc2['ID_caller'].astype(int)
inc2['opened_by']=inc2['opened_by'].astype(int)
inc2['updated_by']=inc2['updated_by'].astype(int)
inc2['location']=inc2['location'].astype(int)
inc2['Category_id']=inc2['Category_id'].astype(int)
inc2['user_symptom']=inc2['user_symptom'].astype(int)
inc2['Support_group']=inc2['Support_group'].astype(int)


# In[167]:


inc2.drop(['Waitingtime'],axis=1,inplace=True)


# In[169]:


incident1=inc2.copy()


# In[170]:


inc2.to_csv('train_test_pred.csv')


# In[168]:


inc2.info()


# ### Spliting the dataset

# In[171]:


X=incident1.drop("target_impact",axis=1)
y=incident1["target_impact"]


# In[157]:


X.head(2)


# In[158]:


y.head(2)


# In[172]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=10)


# ### Using SMOTE
# ###### To handle im-balanced dataset

# In[173]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[174]:


X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)


# In[175]:


from collections import Counter
print("Before SMOTE :",Counter(y_train))
print("After SMOTE :",Counter(y_train_smote))


# ### Feature Selection

# #### 1. Chi-square Test

# In[176]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[177]:


ordered_rank_feature=SelectKBest(score_func=chi2,k=30)
ordered_feature=ordered_rank_feature.fit(X_train_smote,y_train_smote)
ordered_feature


# In[178]:


df_scores=pd.DataFrame(ordered_feature.scores_,columns=['scores'])
df_columns=pd.DataFrame(X_train.columns)
features_rank=pd.concat([0,df_columns],axis=1)
features_rank.nlargest(30,'scores')


# #### 2. ExtraTreesClassifier

# In[179]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[180]:


model=ExtraTreesClassifier()
model.fit(X_train_smote,y_train_smote)


# In[181]:


print(model.feature_importances_)


# In[182]:


plt.figure(figsize=(20,10))
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(20).plot(kind='barh')
plt.show()


# #### 3. Mutual Information Classifier

# In[183]:


from sklearn.feature_selection import mutual_info_classif


# In[184]:


mutual_info=mutual_info_classif(X_train_smote,y_train_smote)


# In[185]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[186]:


#let's plot the ordered mutual_info values per feature
mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 4))


# In[187]:


#Now we Will select the  top 10 important features
sel_top_cols = SelectKBest(mutual_info_classif, k=10)
sel_top_cols.fit(X_train_smote, y_train_smote)
X_train.columns[sel_top_cols.get_support()]


# #### 4. Decision Tree Classifier 

# In[188]:


from sklearn.tree import DecisionTreeClassifier


# In[189]:


model= DecisionTreeClassifier()
model.fit(X_train_smote,y_train_smote)


# In[191]:


feature = pd.Series(model.feature_importances_)
feature.index = X_train.columns
feature.sort_values(ascending=False)


# In[192]:


feature.sort_values(ascending=False).plot.bar(figsize=(15, 4))


# In[252]:


X_train_new= X_train_smote[['opened_by','location','ID_caller','Category_id','ID']]
X_test_new= X_test[['opened_by','location','ID_caller','Category_id','ID']]


# ### 1. Random Forest Classifier

# In[255]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(max_depth=27)
model_rf.fit(X_train_new,y_train_smote)


# In[256]:


# Predicting the model
Y_predict_rf = model_rf.predict(X_test_new)


# In[257]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics


# In[258]:


# Finding accuracy, precision, recall and confusion matrix
from sklearn.metrics import classification_report
print(accuracy_score(y_test,Y_predict_rf))
print(classification_report(y_test,Y_predict_rf))


# In[259]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,Y_predict_rf)


# In[260]:


print("Train Accuracy:",model_rf.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",model_rf.score(X_test_new, y_test)*100)


# ### 2. Neural Network

# In[261]:


from sklearn.neural_network import MLPClassifier


# In[262]:


model_mlp = MLPClassifier()
model_mlp.fit(X_train_new,y_train_smote)


# In[263]:


# Predicting the model
Y_predict_mlp = model_mlp.predict(X_test_new)


# In[264]:


# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,Y_predict_mlp))
print(classification_report(y_test,Y_predict_mlp))


# In[265]:


print("Train Accuracy:",model_mlp.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",model_mlp.score(X_test_new, y_test)*100)


# ### 3. Support Vector Classifier

# In[266]:


from sklearn.svm import SVC
# Training the model
rbf_svc = SVC(kernel='rbf',C=10,gamma=0.1).fit(X_train_new,y_train_smote)
# Predicting the model
Y_predict_svm = rbf_svc.predict(X_test_new)


# In[267]:


# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,Y_predict_svm))


# In[268]:


print(classification_report(y_test,Y_predict_svm))


# In[269]:


print("Train Accuracy:",rbf_svc.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",rbf_svc.score(X_test_new, y_test)*100)


# ### 4. KNN

# In[270]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_new,y_train_smote)


# In[271]:


pred = knn.predict(X_test_new)


# In[272]:


print(confusion_matrix(y_test,pred))


# In[273]:


print(classification_report(y_test,pred))


# In[274]:


print("Train Accuracy:",knn.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",knn.score(X_test_new, y_test)*100)


# ### 5. Decision Tree Classifier

# In[275]:


from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree


# In[276]:


#Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(X_train_new,y_train_smote)


# In[277]:


#PLot the decision tree
tree.plot_tree(model);


# In[278]:


#Predicting on test data
preds = model.predict(X_test_new) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[279]:


# Accuracy 
np.mean(preds==y_test)


# In[280]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[281]:


model_gini.fit(X_train_new, y_train_smote)


# In[282]:


#Prediction and computing the accuracy
pred=model.predict(X_test_new)
np.mean(preds==y_test)


# In[283]:


print("Train Accuracy:",model_gini.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",model_gini.score(X_test_new, y_test)*100)


# ### 6. Gausian NB

# In[284]:


from sklearn.naive_bayes import GaussianNB


# In[285]:


#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train_new, y_train_smote)

#Predict the response for test dataset
y_pred = gnb.predict(X_test_new)


# In[286]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[287]:


print(classification_report(y_test,y_pred))


# In[288]:


print("Train Accuracy:",gnb.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",gnb.score(X_test_new, y_test)*100)


# ### Voting Classifier

# In[289]:


from sklearn.ensemble import VotingClassifier
nb = GaussianNB()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=101)


# In[290]:


# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('GaussianNB', nb),
('K Nearest Neighbours', knn),
('Classification Tree', dt)]


# In[294]:


# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
  clf.fit(X_train_new, y_train_smote)
  y_pred = clf.predict(X_test_new)
  print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))


# In[295]:


vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train_new, y_train_smote)
y_pred = vc.predict(X_test_new)
print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))


# In[296]:


print("Train Accuracy:",vc.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",vc.score(X_test_new, y_test)*100)


# ### Bagging Classifier

# In[297]:


from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(random_state=101)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train_new, y_train_smote)
y_pred = bc.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))


# In[298]:


print("Train Accuracy:",bc.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",bc.score(X_test_new, y_test)*100)


# ### AdaBoost Classifier

# In[299]:


from sklearn.ensemble import AdaBoostClassifier
dt = DecisionTreeClassifier(random_state=101)
abc = AdaBoostClassifier(base_estimator=dt, n_estimators=300)
abc.fit(X_train_new, y_train_smote)
y_pred = bc.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))


# In[300]:


print("Train Accuracy:",abc.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",abc.score(X_test_new, y_test)*100)


# ### XgBoost Classifier

# In[301]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train_new, y_train_smote)
y_pred = xgb.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)


# In[302]:


print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))


# In[303]:


print("Train Accuracy:",xgb.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",xgb.score(X_test_new, y_test)*100)


# ### Hyperparameter Tuning

# ### Grid Search CV 

# In[304]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
gbm_param_grid = {'learning_rate': [0.05],'n_estimators': [50, 100, 150],'max_depth': [6, 7, 8]}
gbm = xgb.XGBClassifier()
grid_cv = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, cv=4, verbose=1, n_jobs=-1)
grid_cv.fit(X_train_new, y_train_smote)
print("Best parameters found: ",grid_cv.best_params_)


# In[305]:


print("Best Score found: ", np.sqrt(np.abs(grid_cv.best_score_)))


# ### Randomised Forest

# In[316]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
gbm_param_grid = {'learning_rate': np.arange(0.05,1.05,.05),'n_estimators': [100, 200, 300],'max_depth': range(1,10)}
gbm = xgb.XGBClassifier()
randomized_cv = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, n_iter=25,random_state=101, cv=4, verbose=1, n_jobs=-1)
randomized_cv.fit(X_train_new, y_train_smote)
print("Best parameters found: ",randomized_cv.best_params_)


# In[317]:


print("Best Score found: ", np.sqrt(np.abs(randomized_cv.best_score_)))


# In[ ]:




