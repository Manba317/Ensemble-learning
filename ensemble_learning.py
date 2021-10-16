#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
# import matplotlib as plt
from sklearn.metrics import accuracy_score


def load_loan_pred(path_train):
    # reading the dataset
    df = pd.read_csv(path_train)

    print(df.head())
    print(df.describe())
    print(df.isnull().sum())

    # filling missing values
    # pre-process according to https://www.jianshu.com/p/3efc09ef4369
    df['Gender'].fillna(df['Gender'].value_counts().idxmax(), inplace=True)
    df['Married'].fillna(df['Married'].value_counts().idxmax(), inplace=True)
    df['Dependents'].fillna(df['Dependents'].value_counts().idxmax(), inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].value_counts().idxmax(), inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].mean(skipna=True), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].value_counts().idxmax(), inplace=True)
    df['LoanAmount_log'] = np.log(df['LoanAmount'])
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'])
    # df['LoanAmount_log'].hist(bins=20)

    # print(df.isnull().sum())
    # 将非数字类型数据转为数字类型
    from sklearn.preprocessing import LabelEncoder
    var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    print(df.head())
    # 都是一些值为非数值的特征，将这些特征用one-hot编码来表示
    fields = ['Credit_History', 'Education', 'Married', 'Self_Employed', 'Property_Area','Loan_Status']

    df = df[fields]
    print(df.head())

    # split dataset into train and test
    train, test = train_test_split(df, test_size=0.3, random_state=0)

    x_train = train.drop('Loan_Status', axis=1)
    y_train = train['Loan_Status']

    x_test = test.drop('Loan_Status', axis=1)
    y_test = test['Loan_Status']
    print(x_train.head())
    # create dummies
    # 将数据转为one-hot编码，主要是对非数值特征处理
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)
    print(x_train.head())
    return  x_train, y_train, x_test, y_test

# 多个分类起投票得出最后预测结果，hard投票指每个分类器各自先得出预测结果再投票
def exp_voting(x_train,y_train,x_test,y_test):
    model1 = LogisticRegression(random_state=1)
    model2 = DecisionTreeClassifier(random_state=1)
    model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
    model.fit(x_train,y_train)
    model.score(x_test,y_test)

# 软投票的做法
def exp_avg(x_train,y_train,x_test):
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()
    model3= LogisticRegression()

    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)

    pred1=model1.predict_proba(x_test)
    pred2=model2.predict_proba(x_test)
    pred3=model3.predict_proba(x_test)

    finalpred=(pred1+pred2+pred3)/3


def exp_avg_weighted(x_train,y_train,x_test,y_test):
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()
    model3= LogisticRegression()

    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)

    pred1=model1.predict_proba(x_test)
    pred2=model2.predict_proba(x_test)
    pred3=model3.predict_proba(x_test)

    y_pred=(pred1*0.3+pred2*0.3+pred3*0.4)
    score = accuracy_score(y_test, y_pred)
    print(score)


# 一半数据来训练不同的弱分类器，将弱分类的输出结合另一半数据再训练一个模型
# 为了解决数据不足的问题，采用k折交叉训练
def Stacking(model,train,y,test,n_fold):
    # StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)

    for train_indices,val_indices in folds.split(train,y.values):
       x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
       y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
       model.fit(X=x_train,y=y_train)
       train_pred=np.append(train_pred,model.predict(x_val))
       test_pred=np.append(test_pred,model.predict(test))

    return test_pred.reshape(-1,1),train_pred


def exp_stacking(x_train,y_train,x_test,y_test):
    model1 = DecisionTreeClassifier(random_state=1)
    test_pred1, train_pred1 = Stacking(model=model1, n_fold=10, train=x_train, test=x_test, y=y_train)
    train_pred1 = pd.DataFrame(train_pred1)
    test_pred1 = pd.DataFrame(test_pred1)

    model2 = KNeighborsClassifier()
    test_pred2, train_pred2 = Stacking(model=model2, n_fold=10, train=x_train, test=x_test, y=y_train)
    train_pred2 = pd.DataFrame(train_pred2)
    test_pred2 = pd.DataFrame(test_pred2)

    df = pd.concat([train_pred1, train_pred2], axis=1)
    df_test = pd.concat([test_pred1, test_pred2], axis=1)

    model = LogisticRegression(random_state=1)
    model.fit(df, y_train)
    score = model.score(df_test, y_test)
    print(score)


def exp_bagging(x_train,y_train,x_test,y_test):
    from sklearn.ensemble import BaggingClassifier
    from sklearn import tree
    model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(score)


def exp_bagging_regress(x_train,y_train,x_test,y_test):
    from sklearn.ensemble import BaggingRegressor
    model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
    model.fit(x_train, y_train)
    model.score(x_test,y_test)


def exp_adaboost(x_train,y_train,x_test,y_test):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=1)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(score)


def exp_adaboost_regress(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import AdaBoostRegressor
    model = AdaBoostRegressor()
    model.fit(x_train, y_train)
    model.score(x_test, y_test)


def exp_gbm(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.01, random_state=1)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(score)


def exp_gbm_regress(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    model.score(x_test, y_test)


def exp_xgboost(x_train, y_train, x_test, y_test):
    import xgboost as xgb
    model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(score)


def exp_xgboost_regress(x_train, y_train, x_test, y_test):
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
    model.score(x_test, y_test)


def exp_lgb(x_train, y_train, x_test, y_test):
    import lightgbm as lgb
    train_data = lgb.Dataset(x_train, label=y_train)
    # define parameters
    params = {'learning_rate': 0.001}
    model = lgb.train(params, train_data, 100)
    # lightgbm 没有model.score方法
    y_pred = model.predict(x_test)
    for i in range(0, 185):
        if y_pred[i] >= 0.5:
            y_pred[i] = 1
    else:
        y_pred[i] = 0
    score = accuracy_score(y_test, y_pred)
    print(score)


def exp_lgb_regress(x_train, y_train, x_test, y_test):
    import lightgbm as lgb
    train_data = lgb.Dataset(x_train, label=y_train)
    params = {'learning_rate': 0.001}
    model = lgb.train(params, train_data, 100)
    y_pred = model.predict(x_test)
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(y_pred, y_test) ** 0.5


def exp_catboost(x_train, y_train, x_test, y_test):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    # categorical_features_indices = np.where(df.dtypes != np.float)[0]
    # cat_features指定那几个是类别型特征，类别型特征是离散的，xgboost不可以处理类别型特征，会把其当为数值特征计算，catboost可以
    model.fit(x_train, y_train, cat_features=([0, 1, 2, 3, 4, ]), eval_set=(x_test, y_test))
    score = model.score(x_test, y_test)
    print(score)


def exp_catboost_regress(x_train, y_train, x_test, y_test):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    # categorical_features_indices = np.where(df.dtype s != np.float)[0]
    model.fit(x_train, y_train, cat_features=([0, 1, 2, 3, 4, 10]), eval_set=(x_test, y_test))
    model.score(x_test, y_test)


def main():
    path_train = './train_loan_pred.csv'
    [x_train, y_train, x_test, y_test] = load_loan_pred(path_train)
    # voting/avging
    exp_voting(x_train,y_train,x_test,y_test)
    # exp_avg(x_train, y_train, x_test)
    # exp_avg_weighted(x_train, y_train, x_test)
    # stacking
    # exp_stacking(x_train, y_train, x_test, y_test)
    # bagging
    # exp_bagging(x_train, y_train, x_test, y_test)
    # boosting
    # exp_adaboost(x_train, y_train, x_test, y_test)
    # exp_gbm(x_train, y_train, x_test, y_test)
    # exp_xgboost(x_train, y_train, x_test, y_test)
    exp_lgb(x_train, y_train, x_test, y_test)
    # exp_catboost(x_train, y_train, x_test, y_test)

    # bagging/boosting regression
    # exp_bagging_regress(x_train, y_train, x_test, y_test)
    # exp_adaboost_regress(x_train, y_train, x_test, y_test)
    # exp_gbm_regress(x_train, y_train, x_test, y_test)
    # exp_xgboost_regress(x_train, y_train, x_test, y_test)
    # exp_lgb_regress(x_train, y_train, x_test, y_test)
    # exp_catboost_regress(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('whole time: {:.2f} min'.format(t_all / 60.))
