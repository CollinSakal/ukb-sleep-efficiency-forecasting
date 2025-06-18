# Script for creating the final data frames

# Libraries
library(tidyverse)

# Survey data
df_survey <- read_csv('data/derived/df-nonacc.csv') %>% select(-closest_visit)

# Label data
df_labels <- read_csv('data/derived/df-labels.csv') 

# Wearable feature dfs
df_wrb_4hrs <- read_csv('data/derived/df-wearables-4hrs.csv', show_col_types = FALSE)
df_wrb_8hrs <- read_csv('data/derived/df-wearables-8hrs.csv', show_col_types = FALSE)

# Combine 
df_final_4hrs <- inner_join(df_labels, df_wrb_4hrs, by='eid') %>% inner_join(., df_survey, by='eid') 
df_final_8hrs <- inner_join(df_labels, df_wrb_8hrs, by='eid') %>% inner_join(., df_survey, by='eid') 

# Processing
df_final_4hrs <- df_final_4hrs %>%
  mutate(
    slp_eff_avg = rowMeans(select(., slp_eff_t1,slp_eff_t2,slp_eff_t3,slp_eff_t4), na.rm=TRUE),
    slp_dur_avg = rowMeans(select(., slp_dur_t1,slp_dur_t2,slp_dur_t3,slp_dur_t4), na.rm=TRUE)
  )

df_final_8hrs <- df_final_8hrs %>%
  mutate(
    slp_eff_avg = rowMeans(select(., slp_eff_t1,slp_eff_t2,slp_eff_t3,slp_eff_t4), na.rm=TRUE),
    slp_dur_avg = rowMeans(select(., slp_dur_t1,slp_dur_t2,slp_dur_t3,slp_dur_t4), na.rm=TRUE)
  )

# Save
write_csv(df_final_4hrs, 'data/derived/df-final-4hrs.csv')
write_csv(df_final_8hrs, 'data/derived/df-final-8hrs.csv')


