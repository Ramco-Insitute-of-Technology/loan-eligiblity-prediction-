# loan-eligiblity-prediction-
#Importing Necessary libraries:

import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk(''):
for filename in filenames:
print(os.path.join(dirname, filename))

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


#Load and import data
train = pd.read_csv("train_ctrUa4K.csv")
test = pd.read_csv("test_lAUu6dG.csv")
train_original = train.copy()
test_original = test.copy()
train.info()




   








train["Loan_Status"].value_counts(normalize = True).reset_index()



  
#Data Visulisation:
train['Loan_Status'].value_counts(normalize = True).plot.bar()
    




plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize = True).plot.bar(figsize = (20,10), title = 'Gender')
plt.subplot(222)
train['Married'].value_counts(normalize = True).plot.bar(title = 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize = True).plot.bar(title = 'Self Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize = True).plot.bar(title = 'Credit History')



plt.figure(1)
plt.subplot(131)
train['Education'].value_counts(normalize = True).plot.bar(figsize = (20,5), title = 'Education')
plt.subplot(132)
train['Dependents'].value_counts(normalize = True).plot.bar(title = 'Dependents')
plt.subplot(133)
train['Property_Area'].value_counts(normalize = True).plot.bar(title = 'Property area')





plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize = (16,5))
plt.show()


  

train.boxplot(column = 'ApplicantIncome', by = 'Education')
plt.suptitle("")








   




plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize =(20,10))

   
plt.figure(1)
plt.subplot(121)
sns.distplot(train['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot.box(figsize =(20,10))
   

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize =(4,4))

   











  
Married = pd.crosstab(train['Married'],train['Loan_Status'])
Education = pd.crosstab(train['Education'],train['Loan_Status'])
Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])
Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True, figsize = (4,4))
Education.div(Education.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True, figsize = (4,4))
Dependents.div(Dependents.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True, figsize = (4,4))
Self_Employed.div(Self_Employed.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True, figsize = (4,4))

      





         




          







bins = []
bins = [0,2500,4000,6000,81000]
groups = ['Low','Average','High','Very High']
train['Income_bin']=pd.cut(train['ApplicantIncome'], bins, labels = groups)
Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True,)
plt.xlabel("Applicant Income")
plt.ylabel("Percentage")


      











bins = [0,1000,3000,42000]
groups = ['Low','Average','High']
train['Coapplicant_income_bin']=pd.cut(train['CoapplicantIncome'], bins, labels = groups)
Coapplicant_Income = pd.crosstab(train['Coapplicant_income_bin'],train['Loan_Status'])
Coapplicant_Income.div(Coapplicant_Income.sum(1).astype(float), axis = 0).plot(kind ="bar", stacked = True, figsize = (4,4))
plt.xlabel("Co-Applicant Income")
plt.ylabel("Percentage")




     











bins = [0,2500,4000,81000]
groups = ['Low','Average','High']
train['Totalincome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['Total_income_bin']=pd.cut(train['Totalincome'], bins, labels = groups)
Total_Income = pd.crosstab(train['Total_income_bin'],train['Loan_Status'])
Total_Income.div(Total_Income.sum(1).astype(float), axis = 0).plot(kind ="bar", stacked = True, figsize = (4,4))
plt.xlabel("Total Income")
plt.ylabel("Percentage")


      






bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize = (4,4))
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

     










# Exclude non-numeric columns
numeric_train = train.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
matrix = numeric_train.corr()

# Plot heatmap
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=0.8, cmap="YlOrBr", square=True)






      













#Missing value Imputation:
train.isnull().sum()







     



train['Gender'].fillna(train['Gender'].mode()[0], inplace = True)
train['Married'].fillna(train['Married'].mode()[0], inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace = True)
train.isnull().sum()





    







train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


X = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']



X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
x_train.head()
x_train.isnull().values.any()
    



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)

    


#Prediction and Accuracy:

pred_cv = model.predict(x_cv)

accuracy_score(y_cv,pred_cv)

Accuracy is Predicted As 0.8

submission1=pd.read_csv("sample_submission_49d68Cx.csv", index_col=False)

submission1['Loan_Status']=pred_test
submission1['Loan_ID']=test_original['Loan_ID']


