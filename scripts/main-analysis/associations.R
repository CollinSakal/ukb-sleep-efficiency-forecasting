# Main association analyses script

# Libraries
library(glue)
library(mgcv)
library(gratia)
library(ggExtra)
library(ggforce)
library(tidyverse)
library(tidymodels)

# Initializing stuff
y <- 'sleep_efficiency85'

adjustments_full <- glue(
  'age+sex+education+smoking_status+alcohol_freq+',
  'chronotype+sleep_disorder+sleep_medication+cardio_medication+'
)

covariate_names <- c(
  'slp_eff_avg',
  'slp_eff_std', 
  'slp_onset_avg', 
  'slp_onset_std', 
  'slp_offset_avg', 
  'slp_offset_std',
  'slp_dur_avg', 
  'slp_dur_std',
  'M10_act_avg', 
  'M10_act_std',
  'L5_act_avg', 
  'L5_act_std'
)

# # Combine all the necessary data frames and save
# df_target <- read_csv('data/derived/df-final-4hrs.csv') %>% select(eid, sleep_efficiency85)
# df_covs <- read_csv('data/derived/df-associations-covs.csv')
# df_avgs <- read_csv('data/derived/df-final-8hrs.csv') %>% select(eid,all_of(covariate_names))
# df_prebed_4h <- read_csv('data/derived/df-prebed-4hrs.csv') %>% rename(total_acc_4h = total_acc)
# df_prebed_8h <- read_csv('data/derived/df-prebed-8hrs.csv') %>% rename(total_acc_8h = total_acc)
# df_prebed_480m <- read_csv('data/derived/df-prebed-480mins.csv')
# 
# df <-
#   inner_join(df_target, df_covs, by='eid') %>%
#   inner_join(., df_prebed_4h, by='eid') %>%
#   inner_join(., df_prebed_8h, by='eid') %>%
#   inner_join(., df_prebed_480m, by='eid') %>%
#   inner_join(., df_avgs, by='eid')
# 
# write_csv(df, 'data/derived/df-associations-final.csv')

# Read in data, make necessary modifications for analyses
df <- read_csv('data/derived/df-associations-final.csv') %>%
  mutate(
    sex=factor(sex),
    education=factor(education),
    smoking_status=factor(smoking_status),
    alcohol_freq=factor(alcohol_freq),
    chronotype=factor(chronotype),
    sleep_disorder=factor(sleep_disorder),
    sleep_medication=factor(sleep_medication),
    cardio_medication=factor(cardio_medication),
    total_acc_4h = total_acc_4h/100000,     # To report per 100ks
    total_acc_8h = total_acc_8h/100000,     # To report per 100ks
    across(all_of(covariate_names), scale)  # To report per 1-sd increase
  ) %>% 
  drop_na()

# Minute-by-minute analysis ----
minvars <- glue('t{seq(1,480,1)}')

df_results <- tibble(
  term=character(),
  estimate=numeric(),
  std.error=numeric(),
  p.value=numeric()
)

counter <- 1

for(var in minvars){

  formula <- as.formula(glue('{y}~{adjustments_full}+{var}'))
  
  covariate_result <-
    tidy(glm(formula, data=df, family=binomial())) %>%
    filter(str_starts(term, 't')) %>%
    select(term, estimate, std.error, p.value)

  df_results <- bind_rows(df_results, covariate_result)

  print(glue('{counter} of {length(minvars)} completed'))
  counter <- counter+1

}

# Getting the results per 200 unit increase and smoothing
df_results <- df_results %>% 
  mutate(
    conf.low = exp((estimate-1.96*std.error)*200),
    conf.high = exp((estimate+1.96*std.error)*200),
    estimate = exp(estimate*200), # Need to do this last so the CIs don't double exponentiate
  ) 

df_results_smoothed <- df_results %>% 
  mutate(
    estimate = predict(loess(estimate ~ seq(1:nrow(df_results)), span = 0.4, data=df_results)),
    conf.low = predict(loess(conf.low ~ seq(1:nrow(df_results)), span = 0.4, data=df_results)),
    conf.high = predict(loess(conf.high ~ seq(1:nrow(df_results)), span = 0.4, data=df_results))
  )

write_csv(df_results_smoothed, 'results/associations/prebed-480mins-results-smoothed.csv')

# 4-8 hour GAMs ----
formula_4hrs <- as.formula(glue('{y}~{adjustments_full}+s(total_acc_4h)'))
formula_8hrs <- as.formula(glue('{y}~{adjustments_full}+s(total_acc_8h)'))

gam_4hrs <- gam(formula_4hrs, data=df, family=binomial(), method='REML')
gam_8hrs <- gam(formula_8hrs, data=df, family=binomial(), method='REML')

df_plt_4hrs <- smooth_estimates(gam_4hrs, select = 's(total_acc_4h)') %>% 
  add_confint() %>% 
  rename(
    x=total_acc_4h,
    y=.estimate,
    cil=.lower_ci,
    ciu=.upper_ci
  )

df_plt_8hrs <- smooth_estimates(gam_8hrs, select = 's(total_acc_8h)') %>% 
  add_confint() %>% 
  rename(
    x=total_acc_8h,
    y=.estimate,
    cil=.lower_ci,
    ciu=.upper_ci
  )

# 4-days sleep GAMs ----
# Same df as above just w/o scaling
df <- read_csv('data/derived/df-associations-final.csv') %>%
  mutate(
    sex=factor(sex),
    education=factor(education),
    smoking_status=factor(smoking_status),
    alcohol_freq=factor(alcohol_freq),
    chronotype=factor(chronotype),
    sleep_disorder=factor(sleep_disorder),
    sleep_medication=factor(sleep_medication),
    cardio_medication=factor(cardio_medication),
    total_acc_4h = total_acc_4h/100000,     # To report per 100ks
    total_acc_8h = total_acc_8h/100000,     # To report per 100ks
    M10_act_avg = M10_act_avg/100000,       # To report per 100ks
    M10_act_std = M10_act_std/100000,       # To report per 100ks
    L5_act_avg = L5_act_avg/100000,         # To report per 100ks
    L5_act_std = L5_act_std/100000,         # To report per 100ks
  ) %>%
  drop_na()

# .... Mean metrics 
formula_slpdur <- as.formula(glue('{y}~{adjustments_full}+s(slp_dur_avg)'))
formula_slpeff <- as.formula(glue('{y}~{adjustments_full}+s(slp_eff_avg)'))
formula_slpons <- as.formula(glue('{y}~{adjustments_full}+s(slp_onset_avg)'))
formula_slpoff <- as.formula(glue('{y}~{adjustments_full}+s(slp_offset_avg)'))

gam_slpdur <- gam(formula_slpdur, data=df, family=binomial(), method='REML')
gam_slpeff <- gam(formula_slpeff, data=df, family=binomial(), method='REML')
gam_slpons <- gam(formula_slpons, data=df, family=binomial(), method='REML')
gam_slpoff <- gam(formula_slpoff, data=df, family=binomial(), method='REML')

df_plt_slpdur <- smooth_estimates(gam_slpdur, select = 's(slp_dur_avg)') %>% add_confint() 
df_plt_slpeff <- smooth_estimates(gam_slpeff, select = 's(slp_eff_avg)') %>% add_confint() 
df_plt_slpons <- smooth_estimates(gam_slpons, select = 's(slp_onset_avg)') %>% add_confint() 
df_plt_slpoff <- smooth_estimates(gam_slpoff, select = 's(slp_offset_avg)') %>% add_confint() 

# .... SD metrics 
formula_slpdur <- as.formula(glue('{y}~{adjustments_full}+s(slp_dur_std)'))
formula_slpeff <- as.formula(glue('{y}~{adjustments_full}+s(slp_eff_std)'))
formula_slpons <- as.formula(glue('{y}~{adjustments_full}+s(slp_onset_std)'))
formula_slpoff <- as.formula(glue('{y}~{adjustments_full}+s(slp_offset_std)'))

gam_slpdur <- gam(formula_slpdur, data=df, family=binomial(), method='REML')
gam_slpeff <- gam(formula_slpeff, data=df, family=binomial(), method='REML')
gam_slpons <- gam(formula_slpons, data=df, family=binomial(), method='REML')
gam_slpoff <- gam(formula_slpoff, data=df, family=binomial(), method='REML')

df_plt_slpdur <- smooth_estimates(gam_slpdur, select = 's(slp_dur_std)') %>% add_confint() 
df_plt_slpeff <- smooth_estimates(gam_slpeff, select = 's(slp_eff_std)') %>% add_confint() 
df_plt_slpons <- smooth_estimates(gam_slpons, select = 's(slp_onset_std)') %>% add_confint() 
df_plt_slpoff <- smooth_estimates(gam_slpoff, select = 's(slp_offset_std)') %>% add_confint() 

# 4 days acc GAMS ----
# .... Mean metrics 
formula_l5act <- as.formula(glue('{y}~{adjustments_full}+s(L5_act_avg)'))
formula_m10act <- as.formula(glue('{y}~{adjustments_full}+s(M10_act_avg)'))

gam_l5act <- gam(formula_l5act, data=df, family=binomial(), method='REML')
gam_m10act <- gam(formula_m10act, data=df, family=binomial(), method='REML')

df_plt_l5act <- smooth_estimates(gam_l5act, select = 's(L5_act_avg)') %>% add_confint() 
df_plt_m10act <- smooth_estimates(gam_m10act, select = 's(M10_act_avg)') %>% add_confint() 

# .... SD metrics 
formula_l5act <- as.formula(glue('{y}~{adjustments_full}+s(L5_act_std)'))
formula_m10act <- as.formula(glue('{y}~{adjustments_full}+s(M10_act_std)'))

gam_l5act <- gam(formula_l5act, data=df, family=binomial(), method='REML')
gam_m10act <- gam(formula_m10act, data=df, family=binomial(), method='REML')

df_plt_l5act <- smooth_estimates(gam_l5act, select = 's(L5_act_std)') %>% add_confint() 
df_plt_m10act <- smooth_estimates(gam_m10act, select = 's(M10_act_std)') %>% add_confint() 