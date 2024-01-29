#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import joblib
from tqdm import tqdm
import neptune
import os
import logging
from datetime import datetime
from pyCompare import blandAltman
import time

# %%
logging.basicConfig(level=logging.INFO)

df = pd.read_csv('CPET_master_file.csv',decimal=',')
cpet_duration = df['CPET_duration'][~df['CPET_duration'].isnull()].values
df = df[~df['CPET_duration'].isnull()][['ID','Age','Weight','Height','BMI']]
# %
df_features = pd.read_csv('ALL_FEATURES.csv',index_col=0)
# dataframe with features
df_all = pd.merge(df,df_features,on='ID')
df_all = df_all.drop('ID',axis=1)

#%%
correlation = df_all.corrwith(pd.Series(cpet_duration))
df_all = df_all[correlation.index[abs(correlation)>0.2]]
#%%
X_list = [
    df_all
]
X_names = [
    'ALL'
]
# %%

class ModelSKL():

    def __init__(self, model_type, **model_params):

        if model_type=='LR':
            mdl = LinearRegression()
        elif model_type=='Lasso':
            mdl = Lasso(**model_params)
        elif model_type=='Ridge':
            mdl = Ridge(**model_params)
        elif model_type=='SVR':
            mdl = SVR(**model_params)
        elif model_type=='RF':
            mdl = RandomForestRegressor(**model_params)
        elif model_type=='BayesianRidge':
            mdl = BayesianRidge(**model_params)
        elif model_type=='ARDRegression':
            mdl = ARDRegression(**model_params)
        elif model_type=='GaussianProcessRegressor':
            mdl = GaussianProcessRegressor(**model_params)
        elif model_type=='GradientBoostingRegressor':
            mdl = GradientBoostingRegressor(**model_params)
        elif model_type=='HuberRegressor':
            mdl = HuberRegressor(**model_params)
        elif model_type=='TheilSenRegressor':
            mdl = TheilSenRegressor(**model_params)
        #TODO inne modele
        self.model = mdl
        self.params = {}
        for key in model_params:
            self.params[key] = model_params[key]

    def fit(self, X_train_scaled, y_train, folder_name, idx, **fitting_params):
        self.model.fit(X_train_scaled, y_train, **fitting_params)

        joblib.dump(self.model, f'{folder_name}/model_{idx}.joblib')
        for key in fitting_params:
            self.params[key] = fitting_params[key]

    def predict(self, X):
        return self.model.predict(X)
    
    def finished_cv(self, folder_name):
        pass



def run_cv(X, y, signals, model_type, model_class, model_params, fitting_params, dataset_version):

    n_splits=36
    kf = KFold(n_splits=n_splits, random_state=123, shuffle=True)

    folder_name = f'Artifacts/Artifacts_{model_type}_{datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]}'
    os.makedirs(folder_name)
    
    y_hat_test_all = np.array([])
    y_test_all = np.array([])
    y_hat_train_all = np.array([])
    y_train_all = np.array([])
    sex_train = np.array([])
    sex_test = np.array([])
    r2_train = []
    r2_test = []
    mse_train = []
    mse_test = []
    mae_train = []
    mae_test = []
    mape_train = []
    mape_test = []   
    corr_train = []
    corr_test = [] 
    
    for idx, (train_index, test_index) in tqdm(enumerate(kf.split(X, y)),total=n_splits):

        X_train, X_test  =  X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test  =  y.iloc[train_index,], y.iloc[test_index,]
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)    

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        correlation = X_train_scaled.corrwith(y_train)
        X_train_scaled = X_train_scaled[correlation.index[abs(correlation)>0.2]]
        X_test_scaled = X_test_scaled[X_train_scaled.columns]
        mdl = model_class(model_type, **model_params)

        mdl.fit(X_train_scaled, y_train, folder_name, idx, **fitting_params)

        y_hat_train = mdl.predict(X_train_scaled)
        y_hat_test = mdl.predict(X_test_scaled)
        y_hat_test_all = np.concatenate([y_hat_test_all, y_hat_test.reshape(-1)])
        y_test_all = np.concatenate([y_test_all, y_test.values])
        y_hat_train_all = np.concatenate([y_hat_train_all, y_hat_train.reshape(-1)])
        y_train_all = np.concatenate([y_train_all, y_train.values])
        r2_train.append(r2_score(y_train, y_hat_train))
        mse_train.append(mean_squared_error(y_train, y_hat_train))
        mae_train.append(mean_absolute_error(y_train, y_hat_train))
        mape_train.append(np.mean(np.abs(y_train-y_hat_train.reshape(-1))/y_train)*100)
        corr_train.append(pearsonr(y_train,y_hat_train.reshape(-1))[0])

        if len(X)>n_splits:
            r2_test.append(r2_score(y_test, y_hat_test))
            mse_test.append(mean_squared_error(y_test, y_hat_test))
            mae_test.append(mean_absolute_error(y_test, y_hat_test))
            mape_test.append(np.mean(np.abs(y_test-y_hat_test.reshape(-1))/y_test)*100)
            corr_test.append(pearsonr(y_test,y_hat_test.reshape(-1))[0])
    if len(X)==n_splits:
        r2_test = [r2_score(y_test_all, y_hat_test_all)]
        mse_test = [mean_squared_error(y_test_all, y_hat_test_all)]
        mae_test = [mean_absolute_error(y_test_all, y_hat_test_all)]
        mape_test = [np.mean(np.abs(y_test_all-y_hat_test_all)/y_test_all)*100]
        corr_test = [pearsonr(y_test_all,y_hat_test_all.reshape(-1))[0]]

    mdl.finished_cv(folder_name)

    axis_scale=250
    fig_y_ypred = plt.figure()
    min_val = (min(np.min(y_test_all), np.min(y_hat_test_all)) // axis_scale) * axis_scale 
    max_val = (max(np.max(y_test_all), np.max(y_hat_test_all)) // axis_scale) * axis_scale + axis_scale
    plt.plot(y_test_all, y_hat_test_all, 'o', alpha=0.4)
    plt.xlabel('True CPET duration [s]')
    plt.ylabel('Predicted CPET duration [s]')
    plt.xticks(np.arange(min_val, max_val+1, axis_scale))
    plt.yticks(np.arange(min_val, max_val+1, axis_scale))
    plt.legend()
    fig_y_ypred.savefig(f'{folder_name}/y_vs_ypred.jpg')

    fig_y_ypred_train = plt.figure()
    min_val = (min(np.min(y_train_all), np.min(y_hat_train_all)) // axis_scale) * axis_scale
    max_val = (max(np.max(y_train_all), np.max(y_hat_train_all)) // axis_scale) * axis_scale + axis_scale
    plt.plot(y_train_all, y_hat_train_all, 'o', alpha=0.2)
    plt.xlabel('True CPET duration [s]')
    plt.ylabel('Predicted CPET duration [s]')
    plt.xticks(np.arange(min_val, max_val+1, axis_scale))
    plt.yticks(np.arange(min_val, max_val+1, axis_scale))
    plt.legend()
    fig_y_ypred_train.savefig(f'{folder_name}/y_vs_ypred_train.jpg')

    blandAltman(y_hat_test_all, y_test_all,
            limitOfAgreement=1.96,
            confidenceInterval=95,
            confidenceIntervalMethod='approximate',
            detrend=None,
            percentage=False,
            savePath=f'{folder_name}//ba.jpg')
    
    res_train = pd.DataFrame({
        'MAPE': mape_train,
        'R2': r2_train,
        'MAE': mae_train,
        'MSE': mse_train,
        'Pearson R': corr_train
    },index=np.arange(len(mape_train)))
    res_test = pd.DataFrame({
        'MAPE': mape_test,
        'R2': r2_test,
        'MAE': mae_test,
        'MSE': mse_test,
        'Pearson R':corr_test
    },index=np.arange(len(mape_test)))

    results  = {'train': res_train, 'test': res_test}
    res_train.to_csv(f'{folder_name}/train_metrics.csv')
    res_test.to_csv(f'{folder_name}/test_metrics.csv')
    pd.DataFrame({
        'y_true': y_test_all,
        'y_pred': y_hat_test_all
        }).to_csv(f'{folder_name}/y_true_pred.csv')
    # Logging to Neptune
    params = mdl.params
    params['dataset_version'] = dataset_version
    params['model_type'] = model_type
    params['n_splits'] = n_splits
    run = neptune.init_run(
        project=os.environ['NEPTUNE_PROJECT'],
        api_token=os.environ['NEPTUNE_API_TOKEN'],
    ) 
    run["parameters"] = params
    for stage in ['train', 'test']:
        for metric in ['MAPE', 'R2', 'MAE', 'MSE', 'Pearson R']:
            run[f'{stage}/{metric}'] = results[stage].mean()[metric]
    run["Artifacts"].upload_files(f"{folder_name}/*")  
    run["sys/tags"].add([signals])
    run.stop()
    logging.info(f"MAPE={results['test'].mean()['MAPE']}")

    # Deleting all artifacts
    for filename in os.listdir(folder_name):
        filepath = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
    os.rmdir(folder_name) 
    plt.close('all')

    return y_hat_test_all, y_test_all, results, fig_y_ypred

#%% LR
dataset_version = '140124_2'
model_params = {}
fitting_params = {}
model_type = 'LR'
for idx in range(len(X_list)):
    X = X_list[idx]
    X_name = X_names[idx]
    y = pd.Series(cpet_duration)
    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
    time.sleep(1)

#% Lasso
dataset_version = '140124_2'
for alpha in [100,25,10,3,2,1.9,1.8,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.3,1.25,1.2,1.15,1.1,1,0.9,0.8,0.5,0.25,0.1,0.1,0.05,1.9,1.8,1.7,1.65]:
    model_params = {'alpha': alpha}
    fitting_params = {}
    model_type = 'Lasso'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        y = pd.Series(cpet_duration)
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
        time.sleep(1)

#% Ridge
dataset_version = '140124_2'
for alpha in [100, 50, 25, 10, 1, 0.1, 0.05, 0.01, 0.005,0.001,0.0001,0.00001]:#
    model_params = {'alpha': alpha}
    fitting_params = {}
    model_type = 'Ridge'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
        time.sleep(1)


#% RF
dataset_version = '140124_2'
model_type = 'RF'

for n_estimators in [10,25,50,100]:
    for max_depth in [2,3,4,5]:
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
            }
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
            time.sleep(1)

#% SVR
dataset_version = '140124_2'
model_type = 'SVR'

for kernel in ['rbf', 'poly']:
    for c in [10, 1, 0.1, 0.01]:
        for epsilon in [1, 0.1, 0.01]:
            model_params = {
                'kernel': kernel,
                'C': c,
                'epsilon': epsilon
                }
            fitting_params = {}
            for idx in range(len(X_list)):
                X = X_list[idx]
                X_name = X_names[idx]
                _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
                time.sleep(1)

#% BayesianRidge

dataset_version = '140124_2'
for alpha1 in [1e-2,1e-4,1e-6]:
    for alpha2 in [1e-2,1e-4,1e-6]:
        for lambda1 in [1e-2,1e-4,1e-6]:
            for lambda2 in [1e-2,1e-4,1e-6]:

                model_params = {
                    'alpha_1': alpha1,
                    'alpha_2': alpha2,
                    'lambda_1': lambda1,
                    'lambda_2': lambda2,
                    }
                fitting_params = {}
                model_type = 'BayesianRidge'
                for idx in range(len(X_list)):
                    X = X_list[idx]
                    X_name = X_names[idx]
                    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#% ARDRegression
dataset_version = '140124_2'
for alpha1 in [1e-2,1e-4,1e-6]:
    for alpha2 in [1e-2,1e-4,1e-6]:
        for lambda1 in [1e-2,1e-4,1e-6]:
            for lambda2 in [1e-2,1e-4,1e-6]:

                model_params = {
                    'alpha_1': alpha1,
                    'alpha_2': alpha2,
                    'lambda_1': lambda1,
                    'lambda_2': lambda2,
                    }
                fitting_params = {}
                model_type = 'ARDRegression'
                for idx in range(len(X_list)):
                    X = X_list[idx]
                    X_name = X_names[idx]
                    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#% GradientBoostingRegressor

dataset_version = '140124_2'
for lr in [0.1, 0.01,1e-3,1e-4]:
    model_params = {
        'learning_rate': lr
        }
    fitting_params = {}
    model_type = 'GradientBoostingRegressor'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#% HuberRegressor
dataset_version = '140124_2'
for epsilon in [1.1, 1.2, 1.3]:
    model_params = {
        'epsilon': epsilon,
        'max_iter':10000
        }
    fitting_params = {}
    model_type = 'HuberRegressor'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#% TheilSenRegressor

dataset_version = '140124_2'
model_params = {
    'max_iter':10000
    }
fitting_params = {}
model_type = 'TheilSenRegressor'
for idx in range(len(X_list)):
    X = X_list[idx]
    X_name = X_names[idx]
    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#% GaussianProcessRegressor
dataset_version = '140124_2'

model_params = {
    }
fitting_params = {}
model_type = 'GaussianProcessRegressor'
for idx in range(len(X_list)):
    X = X_list[idx]
    X_name = X_names[idx]
    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)


#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
import joblib
import tensorflow as tf


            
class ModelMLP():
    def __init__(self, model_type, neurons, batch_norm, dropout, regularization=None, regularization_alpha=0.1):

        if regularization=='l1':
            kernel_regularizer = tf.keras.regularizers.L1(
                    l1=regularization_alpha
                )
        elif regularization=='l2':
            kernel_regularizer = tf.keras.regularizers.L2(
                    l2=regularization_alpha
                )
        else:
            kernel_regularizer = None

        mdl = Sequential()
        for neur in neurons:
            mdl.add(Dense(neur, activation='elu', kernel_regularizer=kernel_regularizer))
            if batch_norm:
                mdl.add(BatchNormalization())
            if dropout>0:
                mdl.add(Dropout(dropout))
        mdl.add(Dense(1, activation='linear'))
        self.model = mdl
        self.params = {
            'neurons': str(neurons)[1:-1],
            'batch_norm': batch_norm,
            'dropout': dropout
        }


    def fit(self, X_train_scaled, y_train, folder_name, idx,learning_rate, epochs, batch_size, decay_steps, decay_rate):
        opt = optimizers.Nadam(learning_rate=learning_rate)
        self.model.compile( 
            optimizer=opt,
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics = [
                metrics.MeanSquaredError(name='mse'),
                metrics.MeanAbsoluteError(name='mae'),
                metrics.MeanAbsolutePercentageError(name='mape'),
                metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        hist = self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_callback], verbose=False)  #, validation_data=(X_validation_scaled, y_valiation) earlystop
        self.model.save(f'{folder_name}/model_{idx}.h5')
        for metric in ['loss', 'mse', 'mae', 'mape', 'rmse', 'lr']:
            plt.figure(f'{folder_name}_{metric}')
            plt.plot(hist.history[metric])
        
        self.params['epochs'] = epochs
        self.params['batch_size'] = batch_size
        self.params['learning_rate'] = learning_rate

        return hist 
    
    def predict(self, X):
        return self.model.predict(X)

    def finished_cv(self, folder_name):
        for metric in ['loss', 'mse', 'mae', 'mape', 'rmse', 'lr']:
            fig = plt.figure(f'{folder_name}_{metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            fig.savefig(f'{folder_name}/{metric}.jpg')



#%% MLP
dataset_version = '140124_2'
model_type = 'MLP'
for neurons in [[25], [50], [50, 25]]:
    for alpha in [1,0.1, 0.01]:
        for learning_rate in [1e-2,1e-3,1e-4]: 
            model_params = {
                'neurons': neurons, 
                'batch_norm': True, 
                'dropout': 0,
                'regularization': 'l1',
                'regularization_alpha': alpha
            }
            fitting_params = {
                'learning_rate' : learning_rate, 
                'epochs': 100, 
                'batch_size': 4, 
                'decay_steps': 10, 
                'decay_rate': 0.95
            }
            for idx in range(len(X_list)):
                X = X_list[idx]
                X_name = X_names[idx]
                _, _, _, _ = run_cv(X, y, X_name, model_type, ModelMLP, model_params, fitting_params, dataset_version)
