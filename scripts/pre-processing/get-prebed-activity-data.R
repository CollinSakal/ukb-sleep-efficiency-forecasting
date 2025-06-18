# Script for getting everyone's p240 mins prebed activity

# Libraries
library(glue)
library(tidyverse)
library(doParallel)

# Initializing stuff
eids <- substr(list.files('data/derived/sleep-summaries'),1,7)

# Mins before bedtime ----
# Register parallel processor
n_cores <- 15
registerDoParallel(n_cores)

# Run parallel processing
get_prebed_activity <- foreach(eid = eids) %dopar% {

  # Libraries
  library(glue)
  library(tidyverse)

  # Helper functions
  source('scripts/utils/prep-acc-file.R')
  source('scripts/utils/prebed-activity.R')

  # Read in the accelerometer file and preprocess it
  dir_acc <- 'C:/Users/Collin/Documents/CityU/Research/ukb-dementia-wearables/data/ukb/accelerometer-files/'
  ext_acc <- '_90004_0_0.csv'
  fpath_acc <- paste0(dir_acc,eid, ext_acc)

  df_acc_raw <- read_csv(fpath_acc, show_col_types=FALSE)
  df_acc <- prep_acc_file(df_acc_raw, eid)

  # Read in the sleep summary file
  dir_slp <- 'data/derived/sleep-summaries/'
  ext_slp <- '-sleep-summary.csv'
  fpath_slp <- paste0(dir_slp,eid,ext_slp)

  df_slp <- read_csv(fpath_slp, show_col_types=FALSE)

  # Get output data
  df_out <- get_prebed_acc(eid=eid, df_slp=df_slp, df_acc=df_acc)

  dir_out <- 'data/derived/prebed-480mins'
  dir_ext <- 'prebed-acc-480mins.csv'
  fpath_out <- glue('{dir_out}/{eid}-{dir_ext}')

  # Save
  write_csv(df_out, fpath_out)

}

# Return to single core computations
stopImplicitCluster()

# Define matrix to store everything
mins <- 480
tibcolnames <- glue('t{seq(1,mins,1)}')
results_mat <- matrix(nrow=length(eids), ncol=length(tibcolnames))

# Read in each acc file and put into the matrix
for(i in 1:length(eids)){

  eid <- eids[i]
  dir <- 'data/derived/prebed-480mins'
  ext <- 'prebed-acc-480mins.csv'
  fpath <- glue('{dir}/{eid}-{ext}')

  results_mat[i,] <- read_csv(fpath, show_col_types = FALSE) %>% pull(count_avg)

  if(i %% 1000 == 0){print(glue('{i} of {length(eids)} processed'))}

}

# Make into one large tibble, add eids
df <- as_tibble(results_mat)
colnames(df) <- rev(tibcolnames)

df <- df %>% mutate(
  eid = eids
)

# Save
write_csv(df, 'data/derived/df-prebed-480mins.csv')
