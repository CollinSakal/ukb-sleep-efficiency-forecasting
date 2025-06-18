# Code for creating a labels data frame

# Imports
import polars as pl
import os

# Get file names
dir = 'data/derived/sleep-summaries'
fnames = os.listdir(dir)
fpaths = [dir + '/' + fname for fname in fnames]

# Initialize dtype schema and columns to select
columns = ['ID', 'date', 'weekday', 'night_number', 'sleeponset', 'wakeup', 'sleep_duration', 'sleep_efficiency']
dtypes = [pl.Int64, pl.Utf8, pl.Utf8, pl.Int64, pl.Float64, pl.Float64, pl.Float64, pl.Float64]

# Initialize empty data frame
df = pl.DataFrame()

# Loop over files, get the last night (row) of sleep data, remove L5/M10 midpoints, vertically stack them
for file in fpaths:
    df = df.vstack(pl.read_csv(file, columns=columns, dtypes=dtypes).tail(1))

# Create target variable using 0.85 sleep efficiency cutoff
df = df.with_columns(
    pl.when(pl.col('sleep_efficiency') < 0.85).then(1).otherwise(0).alias('sleep_efficiency85')
).rename(
    {'ID':'eid'}
)

#df.write_csv('data/derived/df-labels-all.csv')

# Get EIDs from the wearable device feature directory
dir_wrb = 'data/derived/wearable-features-0hrs/'
fnames = os.listdir(dir_wrb)
eids_keep = pl.Series(name = 'eids_keep', values = [f[0:7] for f in fnames]).cast(pl.Int64)

# Exclude extra EIDS from the labels DF
df = df.filter(pl.col('eid').is_in(eids_keep))

# Save the final data frame
df.write_csv('data/derived/df-labels.csv')