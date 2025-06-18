# File for creating the data frames needed to do the pre-bedtime association analyses

# Imports
import os
import polars as pl

# Initialize a counter
iter_0hrs = 0
iter_4hrs = 0
iter_8hrs = 0

# Get data
dir_8hrs = 'data/derived/association-features-8hrs'
dir_4hrs = 'data/derived/association-features-4hrs'

fnames_4hrs = os.listdir(dir_4hrs)
fnames_8hrs = os.listdir(dir_8hrs)

fpaths_4hrs = [dir_4hrs + '/' + fname for fname in fnames_4hrs]
fpaths_8hrs = [dir_8hrs + '/' + fname for fname in fnames_8hrs]

dtypes = [pl.Int64, pl.Float64] # first col eid, rest activity metrics

df_prebed_4hrs = pl.DataFrame()
df_prebed_8hrs = pl.DataFrame()

# Stack all the individual data frames for each forecasting period
for file in fpaths_4hrs:
    df_prebed_4hrs = pl.concat([df_prebed_4hrs, pl.read_csv(file, dtypes=dtypes, null_values=['NA'])], rechunk=True)
    iter_4hrs += 1
    if iter_4hrs % 1000 == 0:
        print(f'{iter_4hrs} files processed')

for file in fpaths_8hrs:
    df_prebed_8hrs = pl.concat([df_prebed_8hrs, pl.read_csv(file, dtypes=dtypes, null_values=['NA'])], rechunk=True)
    iter_8hrs += 1
    if iter_8hrs % 1000 == 0:
        print(f'{iter_8hrs} files processed')

# Save
df_prebed_4hrs.write_csv('data/derived/df-prebed-4hrs.csv')
df_prebed_8hrs.write_csv('data/derived/df-prebed-8hrs.csv')
