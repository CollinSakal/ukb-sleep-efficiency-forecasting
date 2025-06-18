# Script for getting extra features for association analyses

# Libraries
library(doParallel)

# Register parallel processor
n_cores <- 15
registerDoParallel(n_cores)

# Get EIDS for everyone with sleep data 
eids <- substr(list.files('data/derived/sleep-summaries'),1,7)

get_handcrafted_features <- foreach(eid = eids) %dopar% {
  
  # Libraries
  library(tidyverse)
  
  # Helper functions
  source('scripts/utils/prep-acc-file.R')
  source('scripts/utils/get-association-features.R')
  
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
  
  # Get the wearable features for each forecasting period
  out_dfs <- get_association_features(df_slp=df_slp, df_acc=df_acc, eid=eid)
  
  dir_4hrs <- 'data/derived/association-features-4hrs/'
  dir_8hrs <- 'data/derived/association-features-8hrs/'
  dir_othr <- 'data/derived/association-features-others/'
  ext_4hrs <- '-association-features-4hrs.csv'
  ext_8hrs <- '-association-features-8hrs.csv'
  ext_othr <- '-association-features-others.csv'
    
  fpath_4hrs <- paste0(dir_4hrs,eid,ext_4hrs)
  fpath_8hrs <- paste0(dir_8hrs,eid,ext_8hrs)
  fpath_othr <- paste0(dir_othr,eid,ext_othr)
    
  write_csv(out_dfs$df_prebed_4hrs, fpath_4hrs)
  write_csv(out_dfs$df_prebed_8hrs, fpath_8hrs)
  write_csv(out_dfs$df_prebed_others, fpath_othr)
}

# Return to single core computations
stopImplicitCluster()