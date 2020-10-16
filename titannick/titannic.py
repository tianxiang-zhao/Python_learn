import pandas as pd
titanic=pd.read_csv("titanic_train.csv")

print(titanic.head())
#进行统计
print(titanic.describe())
#对age列的空白值进行填充
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())
print(titanic['Age'].describe())
#打印出性别的所有属性
print(titanic['Sex'].unique())
#对性别进行编号
titanic.loc[titanic['Sex']=='male','Sex']=0
titanic.loc[titanic['Sex']=='female','Sex']=1
print(titanic['Embarked'].unique())
titanic['Embarked']=titanic['Embarked'].fillna("S")
titanic.loc[titanic["Embarked"]=='S',"Embarked"]=0
titanic.loc[titanic["Embarked"]=='C',"Embarked"]=1
titanic.loc[titanic["Embarked"]=='Q',"Embarked"]=2

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,StratifiedShuffleSplit, StratifiedKFold
predictors=['Pclass',"Sex","Age","SibSp","Parch","Embarked","Fare"]
alg=LinearRegression()
#进行交叉验证
kf=KFold(titanic.shape[0],random_state=1,n_splits=3)
print(kf)
predictions=[]
# for train,test in kf.split(titanic):
#     train_predictors=(titanic[predictors].iloc[train,:])
#     trian_target=titanic["Survived"].loc[train]
#     alg.fit(train_predictors,trian_target)
#     test_predictors=alg.predict(titanic[predictors].loc[test,:])
#     predictions.append(test_predictors)
# print(predictions)
# import numpy as np
# predictions=np.concatenate(predictions,axis=0)#对数组进行拼接
# predictions[predictions>0.5]=1
# predictions[predictions<=0.5]=0
# accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)
# print(accuracy)
##使用逻辑回归
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
alg=LogisticRegression(random_state=1)
score=sklearn.model_selection.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=3)
print(score.mean())



