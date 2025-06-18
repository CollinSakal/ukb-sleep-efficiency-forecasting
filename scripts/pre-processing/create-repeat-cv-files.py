# Script for creating repeated cross validation files to hold the predictions for the catboost, cnn+lstm, and
#   meta-learner models

# Imports
import random
import polars as pl
from sklearn.model_selection import StratifiedKFold

# Initializing stuff
niter = 1
nfolds = 10
nrepeats = 10

# Get the eids and labels for everyone in the data
eids = pl.read_csv('data/derived/df-labels.csv').get_column('eid')
labels = pl.read_csv('data/derived/df-labels.csv').get_column('sleep_efficiency85')

# Create an empty data frame with all the information we need
df_empty = pl.DataFrame([
    pl.Series('eid', values=eids),
    pl.Series('cnnlstm_pred', values=[None]*len(eids)),
    pl.Series('catboost_pred', values=[None]*len(eids)),
    pl.Series('label',values=labels)
])

# Create one file for each forecasting window
for i in range(nrepeats):

    # Initialize empty column with validation fold indicator
    df_empty = df_empty.with_columns(pl.lit(-1).alias('valid_fold'))

    # Create fold generator and list with valid folds
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random.randint(0, 10000))
    for fold, (train_index, valid_index) in enumerate(skf.split(eids, labels)):
        df_empty[valid_index, 'valid_fold'] = fold

    # Create a file for each fold
    for k in range(nfolds):

        # Create training and validation folds
        tfold = df_empty.filter(pl.col('valid_fold') != k).drop('valid_fold')
        vfold = df_empty.filter(pl.col('valid_fold') == k).drop('valid_fold')

        # Save training dfs
        tfold.write_csv(f'results/repeat-cv/training-preds/train-preds-4hrs/train-preds-4hrs-fold{niter}.csv')
        tfold.write_csv(f'results/repeat-cv/training-preds/train-preds-8hrs/train-preds-8hrs-fold{niter}.csv')

        # Save validation dfs
        vfold.write_csv(f'results/repeat-cv/validation-preds/valid-preds-4hrs/valid-preds-4hrs-fold{niter}.csv')
        vfold.write_csv(f'results/repeat-cv/validation-preds/valid-preds-8hrs/valid-preds-8hrs-fold{niter}.csv')

        # Logging
        print(f'{niter} of {nfolds * nrepeats} files processed')
        niter += 1
