# File for creating the final df wearable data frames for each forecasting period

# Imports
import os
import polars as pl

# Initialize a counter
iter_4hrs = 0
iter_8hrs = 0

# Get file paths
dir_4hrs = 'data/derived/wearable-features-4hrs'
dir_8hrs = 'data/derived/wearable-features-8hrs'

fnames_4hrs = os.listdir(dir_4hrs)
fnames_8hrs = os.listdir(dir_8hrs)

fpaths_4hrs = [dir_4hrs + '/' + fname for fname in fnames_4hrs]
fpaths_8hrs = [dir_8hrs + '/' + fname for fname in fnames_8hrs]

dtypes = [pl.Float64]*65

# Initialize empty data frames for each forecasting period
df_4hrs = pl.DataFrame()
df_8hrs = pl.DataFrame()

# Stack all the individual data frames for each forecasting period
for file in fpaths_4hrs:
    df_4hrs = pl.concat([df_4hrs, pl.read_csv(file, dtypes=dtypes, null_values=['NA'])], rechunk=True)
    iter_4hrs += 1
    if iter_4hrs % 1000 == 0:
        print(f'{iter_4hrs} files processed')

# Save
df_4hrs.write_csv('data/derived/df-wearables-4hrs.csv')

for file in fpaths_8hrs:
    df_8hrs = pl.concat([df_8hrs, pl.read_csv(file, dtypes=dtypes, null_values=['NA'])], rechunk=True)
    iter_8hrs += 1
    if iter_8hrs % 1000 == 0:
        print(f'{iter_8hrs} files processed')

# Save
df_8hrs.write_csv('data/derived/df-wearables-8hrs.csv')