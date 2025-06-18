# File for extracting and saving the individual acc files for the DL models

# Libraries
library(doParallel)

# Initializing stuff
n_cores <- 12

# Registering the cluster
registerDoParallel(n_cores)

# Get EIDS for everyone with sleep data
eids <- substr(list.files('data/derived/sleep-summaries'), 1,7)

# Run parallel processing to get the input files for the CNN+LSTM models
get_cnn_inputs <- foreach(eid = eids) %dopar% {
  
  # Libraries
  library(tidyverse)
  
  # Helper functions
  source('scripts/utils/prep-acc-file.R')
  source('scripts/utils/prep-acc-file-cnn.R')
  
  # Define directories, extensions, and paths for sleep summary and acc files
  dir_slp <- 'data/derived/sleep-summaries/'
  ext_slp <- '-sleep-summary.csv'
  fpath_slp <- paste0(dir_slp,eid,ext_slp)
  
  dir_acc <- 'C:/Users/Collin/Documents/CityU/Research/ukb-dementia-wearables/data/ukb/accelerometer-files/'
  ext_acc <- '_90004_0_0.csv'
  fpath_acc <- paste0(dir_acc,eid, ext_acc)
  
  # Read in the sleep summary and accelerometer files
  df_slp <- read_csv(fpath_slp, show_col_types=FALSE) 
  df_acc_raw <- read_csv(fpath_acc, show_col_types=FALSE)
  df_acc <- prep_acc_file(df_acc_raw, eid)
  
  # Try to process the accelerometer data for the cnn
  cnn_dfs <- try(prep_acc_file_cnn(df_slp, df_acc, eid))
  
  # If the file processed successfully then save it
  if("try-error" %in% class(cnn_dfs)){}else{
    
    # Define outpaths
    fpath_0hrs <- paste0('data/derived/acc-files-0hrs/',eid,'-acc-0hrs.csv')
    fpath_4hrs <- paste0('data/derived/acc-files-4hrs/',eid,'-acc-4hrs.csv')
    fpath_8hrs <- paste0('data/derived/acc-files-8hrs/',eid,'-acc-8hrs.csv')
    
    # Save each file
    write_csv(cnn_dfs$df_acc_0hrs, fpath_0hrs, col_names = FALSE)
    write_csv(cnn_dfs$df_acc_4hrs, fpath_4hrs, col_names = FALSE)
    write_csv(cnn_dfs$df_acc_8hrs, fpath_8hrs, col_names = FALSE)
  }
}

# Return to single core computations
stopImplicitCluster()


