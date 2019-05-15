# %%
import numpy as np

import pandas as pd
import sklearn.svm as svm
from sklearn import preprocessing

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from lightgbm import LGBMModel
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers
from keras.constraints import maxnorm

import sys
import math

np.set_printoptions(threshold=sys.maxsize)

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_log_error(y, y_pred))


def rmsle2(real, predicted):
    sum = 0.0
    for x in range(len(predicted)):
        if predicted[x] < 0 or real[x] < 0:  # check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


def rmsle3(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

pd.set_option('display.max_rows', 500)

svr = svm.SVR()
# %%
trainDat = pd.read_csv('train.csv')
trainDat = trainDat[trainDat['SalePrice'].notnull()]
response = trainDat['SalePrice']
predictor = trainDat.drop(columns='SalePrice')

categoryList = [
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'CentralAir',
    'Electrical',
    'GarageType',
    'PavedDrive',
    'MiscFeature',
    'SaleType',
    'SaleCondition']

ordinalList1 = [
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'HeatingQC',
    'KitchenQual',
    'FireplaceQu',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    ]

ordinalList2 = [
    'BsmtFinType1',
    'BsmtFinType2']
ordinalList3 = [
    'Functional']
ordinalList4 = [
    'GarageFinish']
ordinalList5 = [
    'Fence']

categoryList_HasMissing = predictor[categoryList].columns[
    predictor[categoryList].isnull().any()]
categoryList_NoMissing = predictor[categoryList].columns[
    predictor[categoryList].notnull().all()]

nonCategoryList = predictor.columns[predictor.dtypes != object].tolist()
nonCategoryList.remove('Id')
# predictorNonCat = [x for x in predictor.columns if x not in categoryList]
modelList = nonCategoryList + categoryList + ordinalList1 + ordinalList2 + \
    ordinalList3 + ordinalList4 + ordinalList5

ordinalEnc1CatOrig = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']

ordinalEnc1Cat = []
for i in range(len(ordinalList1)):
    ordinalEnc1Cat.append(ordinalEnc1CatOrig)

ordinalEnc1 = preprocessing.OrdinalEncoder(categories=ordinalEnc1Cat)
ordinalEnc2 = preprocessing.OrdinalEncoder(
    categories=[['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
                ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']])
ordinalEnc3 = preprocessing.OrdinalEncoder(
    categories=[['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']])
ordinalEnc4 = preprocessing.OrdinalEncoder(
    categories=[['Fin', 'RFn', 'Unf', 'NA']])
ordinalEnc5 = preprocessing.OrdinalEncoder(
    categories=[['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']])

preprocess = make_column_transformer(
    (make_pipeline(
        make_column_transformer(
            (make_pipeline(SimpleImputer(strategy='constant'),
                           preprocessing.OneHotEncoder(sparse=False)),
             categoryList_HasMissing),
            (make_pipeline(SimpleImputer(strategy='most_frequent'),
                           preprocessing.OneHotEncoder(sparse=False)),
             categoryList_NoMissing)),
        PCA(n_components=.90, svd_solver='full', whiten=True)), categoryList),
    (make_pipeline(
        SimpleImputer(strategy='most_frequent'), ordinalEnc1), ordinalList1),
    (make_pipeline(
        SimpleImputer(strategy='most_frequent'), ordinalEnc2), ordinalList2),
    (make_pipeline(
        SimpleImputer(strategy='most_frequent'), ordinalEnc3), ordinalList3),
    (make_pipeline(
        SimpleImputer(strategy='most_frequent'), ordinalEnc4), ordinalList4),
    (make_pipeline(
        SimpleImputer(strategy='most_frequent'), ordinalEnc5), ordinalList5),
    (make_pipeline(
        SimpleImputer(strategy='mean'), preprocessing.StandardScaler()),
     nonCategoryList))

# %%
PipelineXGB = make_pipeline(preprocess,
                            XGBRegressor(colsample_bytree=0.9, gamma=0.3,
                                         max_depth=4, min_child_weight=4,
                                         objective='reg:linear', subsample=0.8)
                            )
XGBRegressor()._get_param_names()
model = PipelineXGB.fit(predictor[modelList], response)
resultTrain = PipelineXGB.predict(predictor[modelList])
rmsle(response, resultTrain)
# rmsle2(response, resultTrain)
# rmsle3(response, resultTrain)

# %%
PipelineLGB = make_pipeline(preprocess,
                            LGBMModel(colsample_bytree=0.7, max_depth=4,
                                      min_child_weight=3,
                                      objective='regression', subsample=0.6))
LGBMModel()._get_param_names()
model = PipelineLGB.fit(predictor[modelList], response)
resultTrain = PipelineLGB.predict(predictor[modelList])
rmsle(response, resultTrain)

# %%

# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection


def create_model(optimizer='Adagrad', kernel_initializer='he_normal',
                 activationXlast='relu', dropout_rate=0.0, weight_constraint=0,
                 init_neurons=128, hidden_neurons=256, lr=0.01, decay=0.0):
    model = Sequential()
    model.add(Dense(init_neurons, kernel_initializer=kernel_initializer,
                    input_dim=102,
                    activation=activationXlast)) # ,
                    # kernel_constraint=maxnorm(weight_constraint)))
    # # model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_neurons, kernel_initializer=kernel_initializer,
                    activation=activationXlast))
    model.add(Dense(hidden_neurons, kernel_initializer=kernel_initializer,
                    activation=activationXlast))
    model.add(Dense(hidden_neurons, kernel_initializer=kernel_initializer,
                    activation=activationXlast))
    model.add(Dense(1, kernel_initializer=kernel_initializer,
                    activation='linear'))
    if optimizer == 'Adagrad':
        optimizer = optimizers.adagrad(lr=lr, decay=decay)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer,
                  metrics=['mean_squared_logarithmic_error'])
    return model

# wrap the model using the function you created

PipelineKeras = make_pipeline(
                    preprocess,
                    KerasRegressor(build_fn=create_model,
                                   verbose=0, epochs=1000))
resultTrain
model = PipelineKeras.fit(predictor[modelList], response.values)
resultTrain = PipelineKeras.predict(predictor[modelList])
rmsle(response, resultTrain)

# %%

# GS CV
trainDatPreprocess = preprocess.fit(predictor[modelList])
trainDatPreprocessT = trainDatPreprocess.transform(predictor[modelList])
rmsle_score = make_scorer(rmsle3, greater_is_better=False)
# %%
xgbr = XGBRegressor()
param_grid = {
    'objective': ['reg:linear', 'reg:gamma'],
    'min_child_weight': [4, 5],
    'gamma': [i/10.0 for i in range(3, 6)],
    'subsample': [i/10.0 for i in range(6, 11)],
    'colsample_bytree': [i/10.0 for i in range(6, 11)],
    'max_depth': range(2, 6),
    }


grid_clf = GridSearchCV(
            xgbr, param_grid, n_jobs=3, cv=5, iid=True,
            scoring=rmsle_score)
grid_clf.fit(trainDatPreprocessT, response)
print(grid_clf.best_score_)
print(grid_clf.best_params_)
grid_clf.cv_results_
# %%
lgbm = LGBMModel()
param_grid = {
    'objective': ['regression'],
    'min_child_weight': range(3, 5),
    'subsample': [i/10.0 for i in range(6, 11)],
    'colsample_bytree': [i/10.0 for i in range(6, 11)],
    'max_depth': range(2, 8),
    'num_leaves': range(50, 600, 75)
    }

rmsle_score = make_scorer(rmsle3, greater_is_better=False)
grid_clf = GridSearchCV(lgbm, param_grid, n_jobs=3, cv=5, iid=True,
                        scoring=rmsle_score)
grid_clf.fit(trainDatPreprocessT, response)
print(grid_clf.best_score_)
print(grid_clf.best_params_)

# %%
kerasModel = KerasRegressor(build_fn=create_model, verbose=0, epochs=1000,
                            batch_size=10)
param_grid = {
    # Step 1:
    # 'batch_size': [10, 20, 40, 60, 80, 100],
    # 'epochs': [100, 300, 500, 1000]
    # Step 2:
    # 'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta',
    #               'Adam', 'Adamax', 'Nadam']
    # Step 3: Optimize the optimizer
    # SGD
    # 'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    # 'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # adagrad
    # 'lr': np.arange(0.01, 0.11, 0.02),
    # 'decay': np.arange(0.0, 0.13, 0.03)
    # Step 4:
    # 'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'zero',
    #             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    # Step 5:
    'activationXlast': ['softmax', 'softplus', 'softsign',
                       'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    # Step 6:
    # 'weight_constraint': [1, 2, 3, 4, 5],
    # 'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Step 7:
    # 'init_neurons': range(20, 141, 20),
    # 'hidden_neurons': range(100, 301, 50)
    }
grid_clf = GridSearchCV(kerasModel, param_grid, n_jobs=3, cv=5, iid=True,
                        scoring=rmsle_score)
grid_clf.fit(trainDatPreprocessT, response.values)
print(grid_clf.best_score_)
print(grid_clf.best_params_)
grid_clf.cv_results_

# %%
trainDatPreprocessT.shape

# %%
# ##TEST CODE SUBMISSION
testDat = pd.read_csv('test.csv')

testT = model.predict(testDat[modelList])

outputResult = pd.concat([testDat['Id'], pd.Series(testT, name='SalePrice')],
                         axis=1)
outputResult.to_csv('testResult.csv', index=False)


# %%
