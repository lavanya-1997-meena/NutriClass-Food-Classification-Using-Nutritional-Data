import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Inspecting tha Data:

df=pd.read_csv("F:/Nutrition analysis/data/synthetic_food_dataset_imbalanced - synthetic_food_dataset_imbalanced.csv.csv")
df.head()
df.shape
df.columns=df.columns.str.lower().str.strip()
df.info()
df.describe()

# Find the duplicates and removed it
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# Chect the nan values
df.isna().sum()
# We use apply() with a lambda function to replace missing values in each numeric column with that 
# column's median.
cols=["calories","protein","fat","carbs","sugar","fiber","sodium","cholesterol","glycemic_index","water_content","serving_size"]
df[cols]=df[cols].apply(lambda x: x.fillna(x.median()))
df.isnull().sum()
df.info()
# we check data is balanced or imbalanced
df['food_name'].value_counts()
#A histogram helps us see how the values are distributed.
for col in df.select_dtypes(include='number').columns:
  sns.histplot(df[col])
  plt.title(col)
  plt.show()
# Create a box plot for each numeric column and check the data distribution and possible outliers.
for col in df.select_dtypes(include='number').columns:
  sns.boxplot(df[col])
  plt.title(col)
  plt.show()

# compare each numerical feature with the target variable. 
# The histogram shows how each nutritional feature is distributed across different food classes. 
# This helps us understand which features may be useful for predicting the food name.

feature=["calories","protein","fat","carbs","sugar","fiber","sodium","cholesterol","glycemic_index","water_content","serving_size"]
target="food_name"

for col in feature:
  plt.figure(figsize=(10,5))
  sns.histplot(data=df,x=df[col],hue=target)
  plt.title(f'{col} vs {target}')
  plt.show()

# To count how many times each calorie value occurs for each food name.
df[['calories','food_name']].value_counts()
# Define a function for checks each column for possible outliers, shows the limits, outlier count, 
# outlier values, and skewness.
def cap_outlier(df,columns):

  for col in columns:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1

    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR

    outlier=df.loc[(df[col]<lower)|(df[col]>upper),col]

    print("\nColumn:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("Outlier Count:", len(outlier))
    print("Outlier Values:", outlier.tolist())
    print(df[col].skew())

columns=["calories","protein","fat","carbs","sugar","fiber","sodium","cholesterol","glycemic_index","water_content","serving_size"]

cap_outlier(df,columns)
Outlier_Values=[1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 
                1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763,
                1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 
                1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763,
                1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 511.6274062, 1289.956763, 1289.956763, 
                1289.956763, 1289.956763, 1289.956763, 557.5320257, 1289.956763, 536.3037567, 1289.956763, 1289.956763, 
                1289.956763, 512.8306296, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 
                1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 
                1289.956763, 508.6683226, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 531.3950191, 1289.956763, 
                511.2483217, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 1289.956763, 
                1289.956763, 1289.956763, 1289.956763, 1289.956763, 539.1651604, 1289.956763, 1289.956763, 1289.956763, 
                1289.956763, 533.4459712, 1289.956763, 1289.956763, 523.0823157, 1289.956763]
df[df['calories'].isin(Outlier_Values)][['calories','food_name']].value_counts()
df.groupby('food_name')['calories'].agg(['count','min','max','mean','median']).sort_values('max',ascending=False)
df[df['calories']>600][['food_name','calories','serving_size','protein','fat','carbs','sugar','fiber',
                        'sodium','cholesterol',"glycemic_index",'water_content']]
suspect=df[df['calories']>1200]
print(suspect.shape)
print(suspect.nunique())

# I checked the outliers in all numerical columns. The same 84 rows had unusual values in multiple
# features. After comparing these values with other features,so  I decided not to cap them. I removed those
# 84 rows and created a new DataFrame called df_clean.

df_clean=df[df['calories']!=1289.956763].copy()
df_clean.shape

# we check the data have is any less then 0 :
num_columns=["calories","protein","fat","carbs","sugar","fiber","sodium","cholesterol","glycemic_index","water_content","serving_size"]
(df_clean[num_columns]<0).sum()

df_clean['calories'].plot(kind='box')
df_clean[['calories','food_name']].value_counts(sort='calories',ascending=False).sort_values(ascending=True)
for col in df_clean.select_dtypes(include='number').columns:
  sns.boxplot(df_clean[col])
  plt.title(col)
  plt.show()
df_clean[df_clean['protein']>28][['protein','food_name']].value_counts()
df_clean[df_clean['glycemic_index']<55][['glycemic_index','food_name']].value_counts()
df_clean[df_clean['glycemic_index']<55][['glycemic_index','food_name']].value_counts()
df_clean[df_clean['food_name']=='Steak'][['food_name','carbs','glycemic_index']]
df_clean[df_clean['food_name']=='Steak'][['food_name','carbs','glycemic_index']].describe()
df_clean[df_clean['glycemic_index']<40][['food_name','calories','serving_size','protein','fat','carbs','sugar','fiber','sodium',
                                         'cholesterol',"glycemic_index",'water_content']]
def outlier_list(df,columns):
   for col in columns:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1

    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR

    outlier=df.loc[(df[col]<lower)|(df[col]>upper),[col,'food_name']]

    print("\nColumn:", col)
    print("Outlier Count:", len(outlier))
    print(outlier.value_counts())
    print(outlier['food_name'].value_counts())
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("min:",outlier[col].min())
    print("max:",outlier[col].max())
columns=["calories","protein","fat","carbs","sugar","fiber","sodium","cholesterol","glycemic_index","water_content","serving_size"]
outlier_list(df_clean,columns)
df_clean[df_clean['food_name'].eq('Steak')][['protein','carbs','fat','sugar','calories','water_content','glycemic_index','sodium']]
df_clean[df_clean['food_name'].eq('Steak')][['protein','serving_size','carbs','fat','sugar',
                                             'calories','water_content','glycemic_index','sodium']].describe()
df_clean.columns
cat=['meal_type', 'preparation_method', 'is_vegan', 'is_gluten_free']
for col in cat:
  df_clean.groupby(col)['food_name'].count().plot(kind='bar')
  plt.show()



df_clean['preparation_method'].value_counts().plot(kind='bar')
df_clean['meal_type'].value_counts().plot(kind='bar')

# Compare Two catagorical column which is use chi2_test(meal type,preparation food to food name)
from scipy.stats import chi2_contingency
def chi2_analysis(df,feature_col,target_col):
  # Ho: There is no Diffrence/meal_type & foodname
  # H1: There is a Diffrence/mealtype & foodname

  ctab=pd.crosstab(df[feature_col],df[target_col])

  # Chi-square test
  chi2, p, dof, expected = chi2_contingency(ctab)
  print('Chi: ',chi2)
  print('Pval: ',p)

  alpha=0.05

  if p<alpha:
    print('Reject Ho: There is a Diffrence/mealtype & foodname')
  else:
    print('Reject H1: There is no Diffrence/meal_type & foodname')

    # Countplot
  plt.figure(figsize=(10,5))
  sns.countplot(data=df,x=feature_col,hue=target_col)
  plt.title(f"{feature_col} vs {target_col} Countplot")
  plt.show()

  #stacked_Bar_chart
  plt.figure(figsize=(10,5))
  ctab.plot(kind='bar',stacked=True)
  plt.title(f"{feature_col} vs {target_col} stackedBar_chart")
  plt.show()

chi2_analysis(df_clean,'meal_type','food_name')
chi2_analysis(df_clean,'preparation_method','food_name')


# Model building
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve,precision_score,recall_score,f1_score


# Imbalanced data handaling:


from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline

x=df_clean.drop('food_name',axis=1)                   #put values in x & y
y=df_clean['food_name']

Num_Con_Selected_Features=df_clean[['calories','protein','fat','carbs','sugar','fiber','sodium','cholesterol',
                                    'glycemic_index','water_content','serving_size']]
Cat_selected_features=df_clean[['meal_type']]
cat_col_rank=[['snack', 'breakfast', 'dinner', 'lunch']]

#train,test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

num_con_Transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())  ])
cat_Transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OrdinalEncoder(categories=cat_col_rank))])
Preprocess_step=ColumnTransformer([
    ('num_con_Transformer',num_con_Transformer,Num_Con_Selected_Features.columns),
    ('cat_Transformer',cat_Transformer,Cat_selected_features.columns)
])

# Label encoding for target(y)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

## Model-1

# Logistic Regression Model for Imbalanced Data
ILR_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('model',LogisticRegression())
])
ILR_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=ILR_Model.predict(x_train)
y_pred_test=ILR_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :ILR')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :ILR')
print(classification_report(y_test_encoded,y_pred_test))

# Logistic Regression Model for Imbalanced Data with class weight=balanced
ILRB_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('model',LogisticRegression(class_weight='balanced'))
])
ILRB_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=ILRB_Model.predict(x_train)
y_pred_test=ILRB_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :ILRB')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :ILRB')
print(classification_report(y_test_encoded,y_pred_test))

# Logistic Regression Model for Imbalanced Data with RUS


ILRUS_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('RUS',RandomUnderSampler(random_state=42)),
    ('model',LogisticRegression())
])
ILRUS_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=ILRUS_Model.predict(x_train)
y_pred_test=ILRUS_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :ILRUS')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :ILRUS')
print(classification_report(y_test_encoded,y_pred_test))

## Model-2

# Decision Tree Model

IDT_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('DT_model',DecisionTreeClassifier())
])
IDT_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=IDT_Model.predict(x_train)
y_pred_test=IDT_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :IDT')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :IDT')
print(classification_report(y_test_encoded,y_pred_test))

# Decision Tree Model with balanced

IDTB_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('DT_model',DecisionTreeClassifier(max_depth=12,class_weight='balanced'))
])
IDTB_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=IDTB_Model.predict(x_train)
y_pred_test=IDTB_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :IDTB')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :IDTB')
print(classification_report(y_test_encoded,y_pred_test))

## Model-3

# Random Forest Tree Classifier

IRF_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('RF_model',RandomForestClassifier(n_estimators=10,max_depth=8,class_weight='balanced'))
])
IRF_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=IRF_Model.predict(x_train)
y_pred_test=IRF_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :IRF')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :IRF')
print(classification_report(y_test_encoded,y_pred_test))

## model-4

# XGBoost Tree Classifier

IXGB_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('XGBoost',XGBClassifier(n_estimators=10,max_depth=10))
])
IXGB_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=IXGB_Model.predict(x_train)
y_pred_test=IXGB_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :IXGB')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :IXGB')
print(classification_report(y_test_encoded,y_pred_test))

## Model-5

#SVC
from sklearn.svm import SVC

ISVC_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('SVC',SVC(class_weight='balanced'))
])
ISVC_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=ISVC_Model.predict(x_train)
y_pred_test=ISVC_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :SVC')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :SVC')
print(classification_report(y_test_encoded,y_pred_test))

## Model-6

#KNN
from sklearn.neighbors import KNeighborsClassifier

IKNN_Model=Pipeline(steps=[
    ('preprocess',Preprocess_step),
    ('ROSE',RandomOverSampler(random_state=42)),
    ('KNN',KNeighborsClassifier(n_neighbors=5))
])
IKNN_Model.fit(x_train,y_train_encoded)

# Predict For X_train and X_Test
y_pred_train=IKNN_Model.predict(x_train)
y_pred_test=IKNN_Model.predict(x_test)

# Evaluate For Train and Test
print('Classification Report Train :KNN')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report Test :KNN')
print(classification_report(y_test_encoded,y_pred_test))

#	Gradient Boosting Classifier


from sklearn.ensemble import GradientBoostingClassifier

GB_Model=Pipeline(steps=[
    ('Preprocess_Step',Preprocess_step),
    ('ROSE',RandomOverSampler(random_state=42)),
     ('GradientBoost_classifier',GradientBoostingClassifier(random_state=42))
])
# Fit the Model
GB_Model.fit(x_train,y_train_encoded)


# Predict For X_Train and X_Test

y_pred_train=GB_Model.predict(x_train)
y_pred_test=GB_Model.predict(x_test)

# Evaluate the Model For Both Train And Test
print('Classification Report For Train:GB')
print(classification_report(y_train_encoded,y_pred_train))
print('Classification Report For Test:GB')
print(classification_report(y_test_encoded,y_pred_test))


# save model in pkl file

import pickle
pickle.dump(IXGB_Model,open('Nutriclass_model.pkl','wb'))
pickle.dump(label_encoder, open("food_label_encoder.pkl", "wb"))
