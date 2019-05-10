#%%
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
from keras.layers import Dense, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

import sys
np.set_printoptions(threshold=sys.maxsize)

import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_log_error( y, y_pred ))
def rmsle2(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
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

trainDat = pd.read_csv('train.csv')
trainDat = trainDat[trainDat['SalePrice'].notnull()]
response = trainDat['SalePrice']
predictor = trainDat.drop(columns='SalePrice')

categoryList = ['MSZoning',
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

ordinalList2 = ['BsmtFinType1',
    'BsmtFinType2',]
ordinalList3 = [
    'Functional',       ]
ordinalList4 = [
    'GarageFinish',       ]
ordinalList5 = [
    'Fence',       ]

categoryList_HasMissing = predictor[categoryList].columns[predictor[categoryList].isnull().any()]
categoryList_NoMissing = predictor[categoryList].columns[predictor[categoryList].notnull().all()]



nonCategoryList = predictor.columns[predictor.dtypes != object].drop('Id').tolist()
# predictorNonCat = [x for x in predictor.columns if x not in categoryList]
modelList = nonCategoryList + categoryList + ordinalList1 + ordinalList2 + ordinalList3 + ordinalList4 + ordinalList5


ordinalEnc1CatOrig=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']

ordinalEnc1Cat = []
for i in range(len(ordinalList1)):    
    ordinalEnc1Cat.append(ordinalEnc1CatOrig)


ordinalEnc1 = preprocessing.OrdinalEncoder(categories=ordinalEnc1Cat)
ordinalEnc2 = preprocessing.OrdinalEncoder(categories=[['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']])
ordinalEnc3 = preprocessing.OrdinalEncoder(categories=[['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal']])
ordinalEnc4 = preprocessing.OrdinalEncoder(categories=[['Fin','RFn','Unf','NA']])
ordinalEnc5 = preprocessing.OrdinalEncoder(categories=[['GdPrv','MnPrv','GdWo','MnWw','NA']])

preprocess = make_column_transformer(
    (make_pipeline(
        make_column_transformer((make_pipeline(SimpleImputer(strategy='constant'), preprocessing.OneHotEncoder(sparse=False)), categoryList_HasMissing),
                                (make_pipeline(SimpleImputer(strategy='most_frequent'), preprocessing.OneHotEncoder(sparse=False)), categoryList_NoMissing)),
        PCA(n_components = .90, svd_solver = 'full', whiten=True)), categoryList),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), ordinalEnc1), ordinalList1),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), ordinalEnc2), ordinalList2),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), ordinalEnc3), ordinalList3),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), ordinalEnc4), ordinalList4),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), ordinalEnc5), ordinalList5),
    (make_pipeline(SimpleImputer(strategy='mean'), preprocessing.StandardScaler()), nonCategoryList))   

PipelineXGB = make_pipeline(preprocess,
                        XGBRegressor(colsample_bytree= 0.9, gamma= 0.3, max_depth= 4, min_child_weight= 4, \
                        objective= 'reg:linear', subsample= 0.9))
XGBRegressor()._get_param_names()
model = PipelineXGB.fit(predictor[modelList], response)
resultTrain = PipelineXGB.predict(predictor[modelList])
rmsle(response, resultTrain)
# rmsle2(response, resultTrain)
# rmsle3(response, resultTrain)
PipelineLGB = make_pipeline(preprocess,
                        LGBMModel(colsample_bytree= 0.9, max_depth= 4, min_child_weight= 4, \
                        objective= 'regression', subsample= 0.9))
LGBMModel()._get_param_names()
model = PipelineLGB.fit(predictor[modelList], response)
resultTrain = PipelineLGB.predict(predictor[modelList])
rmsle(response, resultTrain)




# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model():
    model = Sequential()
    model.add(Dense(128, kernel_initializer='normal', input_dim = 101, activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    model.compile(loss='mean_squared_logarithmic_error', optimizer='Adagrad', metrics=['mean_squared_logarithmic_error'])

    return model

# wrap the model using the function you created

PipelineKeras = make_pipeline(preprocess,
                        KerasRegressor(build_fn=create_model,verbose=0))

model = PipelineKeras.fit(predictor[modelList], response.values)   
resultTrain = PipelineKeras.predict(predictor[modelList])
rmsle(response, resultTrain)


#GS CV
trainDatPreprocess = preprocess.fit(predictor[modelList])
trainDatPreprocessT = preprocess.transform(predictor[modelList])

xgbr = XGBRegressor()
param_grid = {
    'objective': ['reg:linear', 'reg:gamma'],
    'min_child_weight':[4,5], 
    'gamma':[i/10.0 for i in range(3,6)],  
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)], 
    'max_depth': range(2,6),
    }

rmsle_score = make_scorer(rmsle3, greater_is_better=False)
grid_clf = GridSearchCV(xgbr, param_grid, n_jobs=2, cv=5, iid=True, scoring=rmsle_score)
grid_clf.fit(trainDatPreprocessT, response)
print(grid_clf.cv_results_)
print(grid_clf.best_score_)
print(grid_clf.best_params_)

lgbm = LGBMModel()
param_grid = {
    'objective': ['regression'],
    'min_child_weight': range(3,5), 
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)], 
    'max_depth': range(2,6),
    }

rmsle_score = make_scorer(rmsle3, greater_is_better=False)
grid_clf = GridSearchCV(lgbm, param_grid, n_jobs=2, cv=5, iid=True, scoring=rmsle_score)
grid_clf.fit(trainDatPreprocessT, response)
print(grid_clf.cv_results_)
print(grid_clf.best_score_)
print(grid_clf.best_params_)

###TEST CODE SUBMISSION
testDat = pd.read_csv('test.csv')

testT = model.predict(testDat[modelList])

outputResult = pd.concat([testDat['Id'], pd.Series(testT, name='SalePrice')], axis = 1)
outputResult.to_csv('testResult.csv', index=False)



# predictorCat = predictor[categoryList].fillna('NaN') 
# oneHotEnc = preprocessing.OneHotEncoder()
# labelEnc = defaultdict(preprocessing.LabelEncoder)
# categoryListLabel = predictorCat.apply(lambda x: labelEnc[x.name].fit(x))
# predictorCat = predictorCat.apply(lambda x: categoryListLabel[x.name].transform(x))
# oneHotEnc = preprocessing.OneHotEncoder(sparse=False).fit(predictorCat)
# predictorCat = oneHotEnc.transform(predictorCat)


# pd.DataFrame(predictorCat)
# predictorCleansed = pd.concat( [predictor[predictorNonCat], pd.DataFrame(predictorCat)], axis=1)

# svrModel = svr.fit(predictorCleansed.drop(columns='Id'), response)
