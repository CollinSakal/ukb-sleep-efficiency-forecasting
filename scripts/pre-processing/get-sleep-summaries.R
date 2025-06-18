# Code for getting the sleep summaries using parallel processing

# Libraries
library(foreach)
library(doParallel)
library(tidyverse)

# Defining the number of cores to use (desktop has 24)
n_cores <- 15

# Registering the cluster
registerDoParallel(n_cores)

# Set directory and get file names for relevant eids
eids <- read_csv('data/derived/df-nonacc.csv') %>% pull(eid) 

# Run the parallel processing code
sleep_summary_computation <- foreach(eid = eids) %dopar% {
  
  # Libraries
  library(tidyverse)
  
  # Helper functions
  source('scripts/utils/prep-acc-file.R')
  source('scripts/utils/sleep-summary.R')
  
  # Define directory and file names
  dir <- 'C:/Users/Collin/Documents/CityU/Research/ukb-dementia-wearables/data/ukb/accelerometer-files/'
  ext <- '_90004_0_0.csv'
  fpath <- paste0(dir, eid, ext)
  
  # Read in the accelerometer file
  df_acc_raw <- read_csv(fpath, show_col_types=FALSE)
  
  # Prep the accelerometer file
  df_acc_processed <- try(prep_acc_file(df=df_acc_raw, eid=eid))
  
  # If the accelerometer file was processed successfully then input it into the
  #   sleep summary function, otherwise stop
  if("try-error" %in% class(df_acc_processed)){}else{

    # Try the sleep summary function
    df_out <- 
      try(
        sleep_summary(
          df_acc_processed,
          interval = 5,       
          minhour = 17, 
          maxtry = 20, 
          sleepstart = 18, 
          sleepend = 11, 
          id_input = eid
        )
      )
    
    # Don't save the sleep summary if an error in the HMM occurred 
    if("try-error" %in% class(df_out)){}else{
      # Don't save the sleep summary if it has <5 rows 
      if(nrow(df_out) < 5){}else{
        setwd("C:/Users/Collin/Documents/CityU/Research/ukb-sleep-forecasting")
        outpath <- paste0("data/derived/sleep-summaries/",eid,"-sleep-summary.csv")
        write.csv(df_out, outpath, row.names = FALSE)
      }
    }
  }
}

# Return to single core computations
stopImplicitCluster()

