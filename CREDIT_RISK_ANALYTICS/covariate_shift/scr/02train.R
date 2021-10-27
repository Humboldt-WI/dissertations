#!/usr/bin/env Rscript
# length(list.files(path = "./", pattern = "\\.Rproj$")) > 0 # to check, whether 

# Source helper_file 
tryCatch(
  error = function(cnd) {
    print(paste0(conditionMessage(cnd), " -- trying different path"))
    source("scr/helpers.R")
    print("Done")
  },
  suppressWarnings(source("helpers.R"))
)

# CLI 
library(docopt)
"Usage:
  02train.R
  02train.R -f --save
  02train.R [options]

Options:
-h --help                     Show this screen.
-f --full                     Run the model for full data
-s --save                     Save the results, the figures
--models=<type>               Run Latt (lat) tree (tree) or (all) [default: all]
--numerr=<1e-n>               Numerical error [default: 0.00001]
--lrate=<n>                   Learning rate [default: 0.01]
--bsize=<n>                   Batch size [default: 128]
--epochs=<n>                  Epochs [default: 25]
--verbose=<bool>              Verbose [default: TRUE]
--modsave=<bool>              Save the lat model [default: FALSE]

" -> doc

opt <- docopt(doc)

# Load (and install) required packages
pkgs = c("docopt", "here", "tidymodels",
         "tidyverse", "tensorflow", "reticulate") 
print(paste0("Loading the following packages: ", paste(pkgs, collapse=", ")))
cat("\n\n")
suppressMessages(load_pkgs(pkgs))

set.seed(4321)

#print(opt) 

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
  reticulate::source_python(here("scr", "feature_configs.py"), 
                            envir = .GlobalEnv) 
}

# Load data from data/cleaned
path <- list.files(here("data/cleaned/"), full.names = T)
data_clean <- purrr::map(path, ~readr::read_csv(.x)) %>%
  purrr::set_names(stringr::str_sub(basename(path), 1, 3))
cat("\n", "data loaded")

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
  numerical_error <- as.numeric(numerical_error)
  learning_rate <- as.numeric(learning_rate)
  batch_size <- as.integer(batch_size)
  epochs <- as.integer(epochs)
  verbose <- ifelse(as.logical(verbose) == TRUE, 2, 0)
  save <- as.logical(save)
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
      dir.create(here("out", "lat_models"))
      lattice_model$save(here(paste0("out/lat_models/", df_name,"_model_new.tf")))
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

if (!opt$full) {
  # Running the model for a single data set
  res <-
    model(df = data_clean$ger,
          models = opt$models,
          numerical_error = opt$numerr,
          learning_rate = opt$lrate,
          batch_size = opt$bsize,
          epochs = opt$epochs,
          verbose = opt$verbose,
          save = opt$modsave,
          df_name = NULL)
} else if (opt$full) {
  # Run the model for all data sets
  res <- data_clean %>%
    purrr::imap(., ~model(df = .x,
                          models = opt$models,
                          numerical_error = opt$numerr,
                          learning_rate = opt$lrate,
                          batch_size = opt$bsize,
                          epochs = opt$epochs,
                          verbose = opt$verbose,
                          save = opt$modsave,
                          df_name = .y)
    ) %>% 
    purrr::map(flatten) %>%
    dplyr::bind_rows() %>% 
    tibble::add_column("data_set" = names(data_clean), 
                       .before = "tr_loss")
}

cat("\n", "The model yields the following results for", ifelse(opt$full, "all data sets", "GER data set"), "\n\n")
print(res)
cat("\n")

# Save results of the experiment
if (opt$save) save_results(res, fname = "results")
cat("\n", "Results saved in /out/res", "\n")