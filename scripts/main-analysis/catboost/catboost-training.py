# Script for tuning the catboost models

# Imports
import random
import optuna
import polars as pl
import statistics as stats

from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Initializing stuff
ntrials = 200

# ----------------------------------------------------------------------------------------------------------------------
#
# FOR 8-HOUR FORECASTING
#
# ----------------------------------------------------------------------------------------------------------------------

def objective_cat(trial):

    # Initialize stuff
    y = 'sleep_efficiency85'
    auc_train, auprc_train, auc_valid, auprc_valid = [], [], [], []
    X_features = pl.read_csv('results/cat-features.csv').get_column('feature_names').to_list()

    # Read in the data frame
    df_8hrs = pl.read_csv('data/derived/df-final-8hrs.csv', null_values=['NA'], infer_schema_length=10000)

    # Suggest hyperparameters
    params = {'loss_function': 'Logloss',
              'eval_metric': 'AUC',
              'verbose': False,
              'random_seed': 19970507,
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.20, log=True),
              'iterations': trial.suggest_int('iterations', 100, 1500),
              'depth': trial.suggest_int('depth', 2, 8),
              'subsample': trial.suggest_float('subsample', 0.75, 1.0),
              'scale_pos_weight': 3.3
              }

    # Get file paths
    nfolds=10

    # Note files indexed starting at 1 not 0
    for fold in range(nfolds):

        # Get validation set eids
        valid_eids = pl.read_csv(f'results/repeat-cv/validation-preds/valid-preds-8hrs/valid-preds-8hrs-fold{fold+1}.csv')\
            .get_column('eid')\
            .to_list()

        # Split the data into training and validation folds
        X_train = df_8hrs.filter(~pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()
        X_valid = df_8hrs.filter(pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()

        y_train = df_8hrs.filter(~pl.col('eid').is_in(valid_eids)).get_column(y).to_numpy()
        y_valid = df_8hrs.filter(pl.col('eid').is_in(valid_eids)).get_column(y).to_numpy()

        # Define CatBoost pools, specify categorical features
        pool_train = Pool(X_train, y_train, feature_names=X_features)
        pool_valid = Pool(X_valid, y_valid, feature_names=X_features)

        # Fit and predict
        model = CatBoostClassifier(**params)
        model.fit(pool_train, eval_set=pool_valid)

        preds_train = model.predict_proba(pool_train)
        preds_valid = model.predict_proba(pool_valid)

        auc_train.append(roc_auc_score(y_train, preds_train[:,1]))
        auc_valid.append(roc_auc_score(y_valid, preds_valid[:,1]))
        auprc_train.append(average_precision_score(y_train, preds_train[:,1]))
        auprc_valid.append(average_precision_score(y_valid, preds_valid[:,1]))

    print(f'Mean train AUC: {stats.mean(auc_train)}, sd: {stats.stdev(auc_train)}')
    print(f'Mean valid AUC: {stats.mean(auc_valid)}, sd: {stats.stdev(auc_valid)}')

    print(f'Mean train AUPRC: {stats.mean(auprc_train)}, sd: {stats.stdev(auprc_train)}')
    print(f'Mean valid AUPRC: {stats.mean(auprc_valid)}, sd: {stats.stdev(auprc_valid)}')

    # Return average validation AUC
    return stats.mean(auc_valid)

# Set up and run the Optuna study
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=ntrials)
print(f'Results.................................................................')
print(f'Mean validation AUC for the best params: {study_cat.best_value}')
print(f'Best params: {study_cat.best_params}')

# Save parameters as a .csv file with performance metrics
df_params_8hrs = pl.DataFrame().with_columns(
    pl.Series(name='model',values=['8hrs']),
    pl.Series(name='best_auc', values=[study_cat.best_value]),
    pl.Series(name='learning_rate', values=[study_cat.best_params['learning_rate']]),
    pl.Series(name='iterations', values=[study_cat.best_params['iterations']]),
    pl.Series(name='depth', values=[study_cat.best_params['depth']]),
    pl.Series(name='subsample',values=[study_cat.best_params['subsample']])
)

df_params_8hrs.write_csv(f'results/catboost-params/params-8hrs.csv')

# ----------------------------------------------------------------------------------------------------------------------
#
# FOR 4-HOUR FORECASTING
#
# ----------------------------------------------------------------------------------------------------------------------

def objective_cat(trial):

    # Initialize stuff
    y = 'sleep_efficiency85'
    auc_train, auprc_train, auc_valid, auprc_valid = [], [], [], []
    X_features = pl.read_csv('results/cat-features.csv').get_column('feature_names').to_list()

    # Read in the data frame
    df_4hrs = pl.read_csv('data/derived/df-final-4hrs.csv', null_values=['NA'], infer_schema_length=10000)

    # Suggest hyperparameters
    params = {'loss_function': 'Logloss',
              'eval_metric': 'AUC',
              'verbose': False,
              'random_seed': 19970507,
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.20, log=True),
              'iterations': trial.suggest_int('iterations', 100, 1500),
              'depth': trial.suggest_int('depth', 2, 8),
              'subsample': trial.suggest_float('subsample', 0.75, 1.0),
              'scale_pos_weight': 3.3
              }

    # Get file paths
    nfolds=10

    # Note files indexed starting at 1 not 0
    for fold in range(nfolds):

        # Get validation set eids
        valid_eids = pl.read_csv(f'results/repeat-cv/validation-preds/valid-preds-4hrs/valid-preds-4hrs-fold{fold+1}.csv')\
            .get_column('eid')\
            .to_list()

        # Split the data into training and validation folds
        X_train = df_4hrs.filter(~pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()
        X_valid = df_4hrs.filter(pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()

        y_train = df_4hrs.filter(~pl.col('eid').is_in(valid_eids)).get_column(y).to_numpy()
        y_valid = df_4hrs.filter(pl.col('eid').is_in(valid_eids)).get_column(y).to_numpy()

        # Define CatBoost pools, specify categorical features
        pool_train = Pool(X_train, y_train, feature_names=X_features)
        pool_valid = Pool(X_valid, y_valid, feature_names=X_features)

        # Fit and predict
        model = CatBoostClassifier(**params)
        model.fit(pool_train, eval_set=pool_valid)

        preds_train = model.predict_proba(pool_train)
        preds_valid = model.predict_proba(pool_valid)

        auc_train.append(roc_auc_score(y_train, preds_train[:,1]))
        auc_valid.append(roc_auc_score(y_valid, preds_valid[:,1]))
        auprc_train.append(average_precision_score(y_train, preds_train[:,1]))
        auprc_valid.append(average_precision_score(y_valid, preds_valid[:,1]))

    print(f'Mean train AUC: {stats.mean(auc_train)}, sd: {stats.stdev(auc_train)}')
    print(f'Mean valid AUC: {stats.mean(auc_valid)}, sd: {stats.stdev(auc_valid)}')

    print(f'Mean train AUPRC: {stats.mean(auprc_train)}, sd: {stats.stdev(auprc_train)}')
    print(f'Mean valid AUPRC: {stats.mean(auprc_valid)}, sd: {stats.stdev(auprc_valid)}')

    # Return average validation AUC
    return stats.mean(auc_valid)

# Set up and run the Optuna study
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=ntrials)
print(f'Results.................................................................')
print(f'Mean validation AUC for the best params: {study_cat.best_value}')
print(f'Best params: {study_cat.best_params}')

# Save parameters as a .csv file with performance metrics
df_params_4hrs = pl.DataFrame().with_columns(
pl.Series(name='model',values=['4hrs']),
    pl.Series(name='best_auc', values=[study_cat.best_value]),
    pl.Series(name='learning_rate', values=[study_cat.best_params['learning_rate']]),
    pl.Series(name='iterations', values=[study_cat.best_params['iterations']]),
    pl.Series(name='depth', values=[study_cat.best_params['depth']]),
    pl.Series(name='subsample',values=[study_cat.best_params['subsample']])
)

df_params_4hrs.write_csv(f'results/catboost-params/params-4hrs.csv')

