
# <!---------------------------------------------------------- Notes ----------------------------------------------------------->
#
#
#
# <!---------------------------------------------------------------------------------------------------------------------------->


# Getting started ************************************************************************************************************* #

# Source helper_file 
source("./scr/helpers.R")

# Load (and install) required packages
pkgs <- c("corrr", "skimr", "here", "cowplot","tidymodels",
         "tidyverse", "tensorflow", "reticulate") 
load_pkgs(pkgs)

set.seed(4321)

# For the script to properly run it is sufficient just to activate the 
# accompanying conda env (created through environment.yml)
reticulate::use_condaenv(condaenv = "ba_vwl", required = TRUE)

# Alternatively create a virtualenv with the below code or install the necessary
# python packages manually and set up reticulate accordingly
# py_pkgs = c("numpy", "tensorflow", "tensorflow_lattice")
# virtualenv_create("ba_vwl") # uses the python interpreter of the current session; it should be python=3.8 though
# virtualenv_install(py_pkgs)
# use_virtualenv(virtualenv = "ba_vwl", required = TRUE)

# To check py configuration
reticulate::py_discover_config()

# Importing tensor flow lattice module 
tryCatch(
  error = function(cnd) {
    print(paste0(conditionMessage(cnd), " -- re-try import"))
    tfl <<- reticulate::import("tensorflow_lattice")
    print("Done")
  },
  tfl <<- reticulate::import("tensorflow_lattice")
)

# External .py script to generate feature configs for the model; called within the model
source_ext <- function() {
  reticulate::source_python(here("scripts", "feature_configs.py"), 
                            envir = .GlobalEnv) 
}


# Read in data **************************************************************************************************************** #

data_raw <- load_raw_data(dir = TRUE)


# Preprocess ****************************************************************************************************************** #

# Variable selection ---------------------------------------------------------------------------------------------------------- #

# Correlation plot for GER
data_raw$ger_raw %>% 
  dplyr::mutate(
    across(where(is.character), as.factor)) %>%
  dplyr::rename(default = credit_risk) %>% 
  corplot()

# Correlation plot for GMC
data_raw$gmc_raw %>% 
  rename(default = SeriousDlqin2yrs) %>% 
  corplot()

# Correlation plot for PAK
data_raw$pak_raw %>%
  mutate(
    across(where(is.character), as.factor)) %>%
  rename(default = X54) %>% 
  corplot()

# Correlation plot for TCD
data_raw$tcd_raw %>% 
  rename(default = default.payment.next.month) %>% 
  corplot()


# Cleaning data --------------------------------------------------------------------------------------------------------------- #

cleaning_data <- function(df, write = FALSE) {
  
  # 
  data_clean <- list()
  
  # ger ----------------------------------------------------------------------------------------------------------------------- #
  
  data_clean[["ger"]] <- df$ger_raw %>%
    dplyr::select(
      -c(employment_duration, personal_status_sex, 
      present_residence, telephone, foreign_worker)
    ) %>%
    dplyr::mutate(
      across(where(is.character), as.factor),
      # Recoding factor levels for applying meaningful monotonicity constraints
      status = recode(status, 
                      "no checking account" = 0, "... < 0 DM" = 1,
                      "0<= ... < 200 DM" = 2, "... >= 200 DM / salary for at least 1 year" = 3
                      ),
      credit_history = recode(credit_history, 
                              "no credits taken/all credits paid back duly" = 0,
                              "all credits at this bank paid back duly" = 1, 
                              "existing credits paid back duly till now" = 2, 
                              "delay in paying off in the past" = 3,
                              "critical account/other credits elsewhere" = 4
                              ),
      savings = recode(savings, 
                       "unknown/no savings account" = 0,
                       "... <  100 DM" = 1,
                       "100 <= ... <  500 DM" = 2,
                       "500 <= ... < 1000 DM" = 3, 
                       "... >= 1000 DM" = 4
                       ),
      installment_rate = recode(installment_rate,
                                "< 20" = 1,
                                "20 <= ... < 25" = 2, 
                                "25 <= ... < 35" = 3, 
                                ">= 35" = 4
                                ),
      number_credits = recode(number_credits, 
                              "1" = 1, "2-3" = 2, 
                              "4-5" = 3, ">= 6" = 4
                              ),
      credit_risk = recode(credit_risk, 
                           "good" = 0, "bad" = 1
                           ),
      across(where(is.factor), as.numeric)
      ) %>%
    dplyr::rename(default = credit_risk)
  
  # gmc ----------------------------------------------------------------------------------------------------------------------- #
  
  modus <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  data_clean[["gmc"]] <- df$gmc_raw %>%
    select(-X1) %>%
    set_names(., 
      c("default", "unsecure_lines", "age", "nr_past_due30", "debt_ratio", 
        "monthly_income", "nr_open_credits", "nr_90days_late", "nr_re_loans", 
        "nr_past_due60", "nr_dependents")
    ) %>%
    mutate(
      age_category = as.factor(case_when(
                               age <= 30 ~ "A",
                               age > 30 & age <= 40 ~ "B",
                               age > 40 & age <= 50 ~ "C",
                               age > 50 & age <= 60 ~ "D",
                               age > 60 & age <= 70 ~ "E",
                               age > 70 ~ "G"))
    ) %>%
    group_by(age_category) %>%
    # Imputing missing values by using mean/mode replacement based on age groups
    mutate(
      monthly_income = replace(monthly_income, is.na(monthly_income), mean(monthly_income, na.rm = TRUE)),
      nr_dependents = replace(nr_dependents, is.na(nr_dependents), modus(nr_dependents))
      ) %>%
    ungroup() %>%
    select(-age_category)
  
  # pak ----------------------------------------------------------------------------------------------------------------------- #
  
  data_clean[["pak"]] <- df$pak_raw %>%
    select(
      X7:X9, X23, X24, X25:X29, 
      X30:X32, X51, X54
    ) %>% 
    set_names(., 
      c("sex", "marriage", "nr_dependants", "income", "income_other", 
        "visa", "master", "diners", "amex", "card_other", "bank_accounts", 
        "bank_accounts_sp", "assets", "age", "default")
    ) %>%
    mutate(
      sex = as.numeric(as.factor(sex)),
      # Create a new var: customer owns credit card y/n
      credit_card = case_when(
                         visa == 1 ~ 1,
                         master == 1 ~ 1,
                         diners == 1 ~ 1,
                         amex == 1 ~ 1,
                         card_other == 1 ~ 1,
                         TRUE ~ 0
                         )
    ) %>%
    select(
      -c(visa, master, diners, amex, card_other)
      ) %>%
    filter(!is.na(sex))
    
  # tcd ----------------------------------------------------------------------------------------------------------------------- #
  
  data_clean[["tcd"]] <- df$tcd_raw %>%
    select(
      -c(ID, MARRIAGE, BILL_AMT1:BILL_AMT6)
      ) %>%
    set_names(., tolower) %>%
    rename(default = default.payment.next.month)# evtl noch eine var raus
  
  # The following function call results in a new subdirectory "data/cleaned", which 
  # contains all above cleaned dfs as .csv files if (write == TRUE) 
  if (write == TRUE) write_cleaned_data(data = data_clean)
    
  return(data_clean)
}

data_clean <- cleaning_data(data_raw, write = FALSE)


# EDA ************************************************************************************************************************* #

# Overview table -------------------------------------------------------------------------------------------------------------- #

# Function gives a brief overview on each data set
overview <- function(){
  n_obs <- function() map(data_clean, nrow) %>% unlist()
  tibble(
    "Data Set" = names(data_clean) %>% toupper(),
    "Default rate" = data_clean %>% 
      map(., select, default) %>% 
      map(., flatten_dbl) %>% 
      map(., mean) %>% unlist(),
    "Nr Obs" = n_obs(),
    "Nr Vars" = map(data_clean, length) %>% unlist(),
    "NA" = c("No", "Yes", "Yes", "No"),
    "Train" = n_obs() * 0.8,
    "Test" = n_obs() * 0.2
  )
}

overview()

## skim ----------------------------------------------------------------------------------------------------------------------- #

# Skim object gives an in-depth overview on individual data sets
my_skim <- skimr::skim_with(
  base = sfl(complete_rate = complete_rate),
  numeric = sfl(p25 = NULL, 
                #p50 = NULL, 
                p75 = NULL, 
                hist = NULL,
                skew = skew),
  character = sfl(empty = NULL,
                  whitespace = NULL)
)

# Show skim overview for all variables
purrr::map(data_clean, my_skim)

## Overview plots ------------------------------------------------------------------------------------------------------------- #

# Ov plot for selected data set and variable
ploting(df = data_clean$gmc, var = monthly_income)

# Ov plots for entire data set
ploting_df(data_clean$gmc, view = TRUE)

# Save ov plot for each data set as pdf in "./results/eda"
invisible(purrr::imap(data_clean, ~ploting_df(., 
                             view = FALSE, 
                             save = TRUE, 
                             df_name = .y))
          )
  

# MODEL *********************************************************************************************************************** #

# for a detailed description of the modeling process see the paper
model <- function(df,
                  models = "all",
                  numerical_error = 1e-5,
                  learning_rate = 0.01,
                  batch_size = 32L,
                  epochs = 25L,
                  verbose = TRUE,
                  save = FALSE,
                  df_name = NULL) {
  ## values
  verbose <- ifelse(verbose == TRUE, 2, 0)
  batch_size <- as.integer(batch_size)
  epochs <- as.integer(epochs)
  res <- list() 
  
  # feature configs ----------------------------------------------------------------------------------------------------------- #
  
  ## Split initial data set into train and test set
  df_split <- rsample::initial_split(df, 
                                     prop = 0.8, 
                                     strata = "default")
  train_data <- rsample::training(df_split)
  test_data <- rsample::testing(df_split)
  
  ## `feature_names` for feature configuration function
  feature_names <- 
    stringr::str_subset(names(df), "default", negate = TRUE)
  
  ## Function to extract and convert covariates to tensors
  extract_features <- function(df, label_name = "default"){
    ls <- list()
    ls[[1]] <- map(select(df, -default), `[`) %>% 
      set_names(., NULL) %>% 
      map(., ~np_array(., dtype = "float64"))
    ls[[2]] <- np_array(df[["default"]], dtype = "float64") 
    names(ls) <- c("xs", "ys")
    return(ls)
  }
  
  ## Extraction
  train_xy <- extract_features(train_data)
  test_xy <- extract_features(test_data)
  
  
  ## Min and max of the dependant variable; 
  ## Small epsilon value from our output_max to make sure we
  ## do not predict values outside of our label bound.
  min_label = min(reticulate::py_to_r(train_xy$ys)) 
  max_label = max(reticulate::py_to_r(train_xy$ys))
  numerical_error_epsilon = numerical_error
  
  # feature configs ----------------------------------------------------------------------------------------------------------- #
  
  ## feature config dict function
  source_ext()
  
  df_name <- ifelse(is.null(df_name), deparse(substitute(df)), df_name) %>% str_sub(.,-3,)
  
  feature_configs <- generate_fconfigs(df_name, 
                                       train_xy$xs, 
                                       feature_names)
  
  # lattice model ------------------------------------------------------------------------------------------------------------- #
  
  if (models == "all" || models == "lat") {
    
    ## model configuration
    lattice_model_config <- tfl$configs$CalibratedLatticeConfig(
      feature_configs = feature_configs,
      output_min = min_label,
      output_max = max_label - numerical_error_epsilon,
      output_initialization = c(min_label, max_label),
      regularizer_configs = c(
        # Torsion regularizer applied to the lattice to make it more linear.
        tfl$configs$RegularizerConfig(name='torsion', l2=1e-2),
        # Globally defined calibration regularizer is applied to all features.
        tfl$configs$RegularizerConfig(name='calib_hessian', l2=1e-2))
    )
    
    lattice_model <- tfl$premade$CalibratedLattice(lattice_model_config)
    
    ## compile
    lattice_model$compile(
      loss = tf$keras$losses$BinaryCrossentropy(),
      metrics = c(tf$keras$metrics$AUC()),
      optimizer = tf$keras$optimizers$Adam(learning_rate)
    )
    
    ## fit
    print(paste0('Fitting the  Lattice Model for ', df_name))
    
    history <- lattice_model$fit(
      train_xy$xs,
      train_xy$ys,
      epochs = epochs,
      batch_size = batch_size,
      verbose = verbose)
    res["l_tr_loss"] <- history$history[[1]][[epochs]]
    res["l_tr_auc"] <- history$history[[2]][[epochs]]
    
    ## evaluate
    print(paste0('Lattice Test Set Evaluation for ', df_name))
    eval <- lattice_model$evaluate(test_xy$xs, test_xy$ys)
    res["l_ts_loss"] <- eval[1]
    res["l_ts_auc"] <- eval[2]
    
    ## save the model
    if(save == TRUE){
      lattice_model$save(here(paste0("results/lat_models/", df_name,"_model_new.tf")))
      print(paste0('Model for ', df_name, ' data set saved'))
    }
  }
  
  # tree model ---------------------------------------------------------------------------------------------------------------- #
  
  if (models == "all" || models == "tree") {
    
    ## model configuration
    train_data <- mutate(train_data, default = factor(default))
    test_data <- mutate(test_data, default = factor(default))
    
    rf_mod <- rand_forest(min_n = 3, trees = 1000) %>% 
      set_engine("ranger") %>% 
      set_mode("classification")
    
    ## fit
    print(paste0('Fitting the  Tree Model for ', df_name))
    
    rf_fit <- 
      rf_mod %>% 
      fit(default ~ ., data = train_data, control_parsnip(verbosity = 2))
    rf_fit
    
    ## evaluate
    print(paste0('Tree Test Set Evaluation for ', df_name))
    
    rf_pred <- function(data_type){
      predict(rf_fit, data_type) %>% 
        bind_cols(predict(rf_fit, data_type, type = "prob")) %>% 
        bind_cols(data_type %>% 
                    select(default))
    }
    
    tree_auc <- function(data_type){
      rf_pred(data_type) %>%
        roc_auc(truth = default, .pred_1) %>%
        pull(.estimate)
    }
    
    res["t_tr_auc"] <- tree_auc(train_data)
    res["t_ts_auc"] <- tree_auc(test_data)
  }  
  
  # fin ----------------------------------------------------------------------------------------------------------------------- #
  
  return(res)
}


# Running the model for a single data set
res_ger <-
  model(df = data_clean$ger,
      numerical_error = 1e-5,
      learning_rate = 0.01,
      batch_size = 128L,
      epochs = 25L,
      verbose = TRUE,
      save = FALSE,
      df_name = NULL)

# Run the model for all data sets
res <- data_clean %>%
  purrr::imap(., ~model(df = .x,
                        models = "all", 
                        numerical_error = 1e-5,
                        learning_rate = 0.01,
                        batch_size = 128,
                        epochs = 25,
                        verbose = TRUE,
                        save = FALSE,
                        df_name = .y)
  ) %>% 
  purrr::map(flatten) %>%
  dplyr::bind_rows() %>% 
  tibble::add_column("data_set" = names(data_clean), 
                     .before = "tr_loss")

# Save results of the experiment
save_results(res, 
             overview(), 
             fname = "results2")

# Load a saved lattice model by supplying the name of the desired data frame
mod_gmc <- load_lat_model(mod = gmc)

# Evaluate the test accuracy of the model
eval_lat_mod(mod = mod_gmc, 
             df = data_clean$gmc)


# Statistical test ************************************************************************************************************ #

# Wilcoxon’s Signed-Rank Test for Matched Pairs
res <- read_rds(here("results/exp_results", "results.RDS"))$res

wilcox.test(res$ts_auc, res$tts_auc, paired = TRUE)


# ***************************************************************************************************************************** #