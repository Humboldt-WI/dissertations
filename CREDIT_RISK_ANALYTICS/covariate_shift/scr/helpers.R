#!/usr/bin/env Rscript

# ***************************************************************************************************************************** #
# ***************************************************  Script installer  ****************************************************** #
# ********************************************  Do not run in interactive mode  *********************************************** #
# ***************************************************************************************************************************** #

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  # stopifnot(args %in% c("--install"))

  installer <- function(pkgs) {
    # Install packages not yet installed
    pkgs <- c("corrr", "skimr", "here", "cowplot","tidymodels",
              "tidyverse", "tensorflow", "reticulate", "docopt")

    installed_pkgs <- pkgs %in% rownames(installed.packages())

    if (any(installed_pkgs == FALSE)) {
      install.packages(pkgs[!installed_pkgs], repos = "https://cran.rstudio.com/")
    }
  }
  if (length(args) > 0) {
    installer()
    cat("\n", "All packages installed", "\n")
  }
}

main()

# ****************************************************  Helper functions  ***************************************************** #


# Package loader function ----------------------------------------------------------------------------------------------------- #

# Loads a list `pkgs`; if a package is not installed, User will be asked to permit 
# installation
load_pkgs <- function(pkgs) {
  # Install packages not yet installed
  installed_pkgs <- pkgs %in% rownames(installed.packages())
  
  if (any(installed_pkgs == FALSE)) {
    try(if (askYesNo("Install missing packages?") == TRUE) {
      install.packages(pkgs[!installed_pkgs])
    } else {
      rlang::abort(paste0("To proceed please install the required packages: ", 
                          pkgs[!installed_pkgs]))
    })
  }
  # Packages loading
  if (sum(installed_pkgs) == length(pkgs)) {
    invisible(lapply(pkgs, library, character.only = TRUE))
  }
}

# Raw Data loader function ---------------------------------------------------------------------------------------------------- #

# If `dir = TRUE` the function loads all data sets from the project data directory; 
# if set FALSE, all data is downloaded from their web sources and then loaded
load_raw_data <- function(dir = TRUE) {
  
  if (dir == TRUE) {
    paths <- c("data/raw/german/german.csv", 
               "data/raw/gmc/cs-training.csv", 
               "data/raw/pak/PAKDD2010_Modeling_Data.txt", 
               "data/raw/taiwanese/UCI_Credit_Card.csv") %>%
      map(., here)
    
    reader <- function(x){
      if(endsWith(x, ".csv")){
        read_csv(x)
      }else{
        read_tsv(x, col_names = FALSE)
      }
    }
    
    map(paths, ~reader(.)) %>% 
      set_names(., c("ger_raw", "gmc_raw", "pak_raw", "tcd_raw")) 
    
  } else {
    
    # Function to load ger data
    load_ger <- function(){
      old <- setwd(here())
      on.exit(setwd(old), add = TRUE)
      
      f <- tempfile()
      download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00522/SouthGermanCredit.zip", f)
      
      d <- tempdir()
      unzip(f, exdir = file.path(d, "GermanCredit"))
      
      cwd <- setwd(d)
      source("GermanCredit/read_SouthGermanCredit.R", local = T)
      setwd(cwd)
      
      ger <- as_tibble(dat)
    }
    
    # Function to load gmc data
    load_gmc <- function(){
      gmc <- read_csv("https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/GiveMeSomeCredit/CreditScoring.csv")
    }
    
    # Function to load pak data
    load_pak <- function(){
      old <- setwd(here())
      on.exit(setwd(old), add = TRUE)
      
      f <- tempfile()
      download.file("https://github.com/JLZml/Credit-Scoring-Data-Sets/raw/master/2.%20PAKDD%202009%20Data%20Mining%20Competition/PAKDD%202010.zip", f)
      
      d <- tempdir()
      unzip(f, exdir = file.path(d, "pak"))
      
      setwd(paste0(d,"/pak"))
      unzip("PAKDD-2010 training data.zip")
      pak <- read_tsv("PAKDD2010_Modeling_Data.txt", col_names = FALSE)
    }
    
    # Function to load tcd data
    load_tcd <- function(){
      tcd <- read_csv("https://raw.githubusercontent.com/serenalwang/shape_constraints_for_ethics/master/credit_default.csv") %>% 
        rename(default.payment.next.month = default)
    }
    
    list("ger_raw" = load_ger(), 
         "gmc_raw" = load_gmc(), 
         "pak_raw"= load_pak(), 
         "tcd_raw" = load_tcd())
  }
  
}

# Plot functions -------------------------------------------------------------------------------------------------------------- #

# Function to plot the correlation of `default` and all independent covariates of input
# data set `df` (does not work for data_raw)
corplot <- function(df){
  df %>%
    dplyr::mutate_if(is.factor, as.numeric) %>%
    corrr::correlate(quiet = TRUE) %>%
    corrr::focus(default) %>%
    dplyr::mutate(rowname = reorder(rowname, default)) %>%
    ggplot(aes(rowname, default)) +
    geom_col(fill = "#4271AE") + coord_flip() +
    #expand_limits(y = c(-1,1)) +
    labs(title = "", y = "Correlation", x = "")
}

# Function to plot a selected variable (`var`) of a data set
ploting <- function(df, 
                    var, 
                    mp = FALSE, 
                    x_lab = NULL) {
  
  if(mp == FALSE){
    var1 <- deparse(substitute(var)) 
    sw <- if_else(length(unique(df[[var1]])) < 50, FALSE, TRUE)
    x_lab <- var1
  } else {
    sw <- if_else(length(unique({{var}})) < 50, FALSE, TRUE)
  }
  
  ggplot(data = df, aes(x = {{var}})) + 
    {
      if (sw){
        geom_histogram(fill = "#FF6347", bins = 10,
                       alpha = 0.8) 
      } else {
        geom_bar(fill = "#4271AE",
                 alpha = 0.8) 
      }
    } +
    theme(panel.grid.minor.y = element_blank(),
          plot.margin = unit(c(1, 1, 1, 1), "cm")) +
    labs(x = x_lab,
         y = NULL) + 
    scale_x_continuous(labels = scales::label_comma(accuracy = 1)) +
    scale_y_continuous(labels = scales::comma)
}

# Function to plot each variable of a data set, including a save option for saving
# the plots in "results/eda"
ploting_df <- function(df, 
                       view = TRUE, 
                       save = FALSE,
                       df_name = NULL) {
  
  if (str_detect(sessionInfo()[[4]], "mac") == FALSE) {
    snk = "NUL"
  } else {
    snk = "/dev/null"
  }
  
  sink(snk)
  plot_list <- imap(df, ~ploting(df = df, 
                                 var = .x, 
                                 mp = TRUE, 
                                 x_lab = .y)) 
  
  if (save == TRUE) {
    if (is.null(df_name)) { 
      df_name <- deparse(substitute(df)) %>% str_sub(.,-3,)
    }
    lp <- length(plot_list)
    s <- seq(1,lp)
    subs <- split(plot_list,
                  ceiling(seq_along(s)/6))
    
    pdf(here("out/plots",
             paste0("ov_plot_", df_name, ".pdf")
    ),
    height = 10,
    #width = 10,
    paper = "a4")
    
    print(map(subs, ~cowplot::plot_grid(plotlist = .x, ncol = 2)))
    dev.off()
  }
  sink()
  
  if (view == TRUE) {
    cowplot::plot_grid(plotlist = plot_list)
  }
}

# Function for percentage bar plot of default for all data sets
default_plot <- function(){
  
  bplot <- function(dataset, df_name){
    dataset %>%
      mutate(default = as.factor(default)) %>%
      ggplot(aes(x = default, y = (..count..)/sum(..count..)), colour = DEFAULT) +
      geom_bar(fill = "#4271AE",
               alpha = 0.8,
               width = 0.5) +
      scale_y_continuous(labels = scales::percent,
                         breaks = seq(0, 1, by = 0.25)) +
      labs(title = "", y = "", x = paste0(df_name)) +
      theme(text = element_text(size=20)) +
      expand_limits(y = c(0, 1)) +
      coord_equal(ratio =2)
  }
  
  plist <- imap(data_clean, ~bplot(dataset = .x, df_name = toupper(.y)))
  cowplot::plot_grid(plist$ger, 
                     plist$gmc + 
                       theme(axis.text.y = element_blank(),
                             #axis.ticks.y = element_blank(),
                             axis.title.y = element_blank()),
                     plist$pak + 
                       theme(axis.text.y = element_blank(),
                             #axis.ticks.y = element_blank(),
                             axis.title.y = element_blank()), 
                     plist$tcd + 
                       theme(axis.text.y = element_blank(),
                             #axis.ticks.y = element_blank(),
                             axis.title.y = element_blank()),
                     nrow = 1,
                     align = "v")
}


# Misc ------------------------------------------------------------------------------------------------------------------------ #

skew <-  function(x, na = TRUE) {
  m3 <- mean((x - mean(x, na.rm = na))^3, na.rm = na)
  skewness <- m3/(sd(x, na.rm = na)^3)
}

# Creates a dir and saves cleaned data
write_cleaned_data <- function(data) {
  dir.create(here("data", "cleaned"))
  purrr::iwalk(.x = data, ~write_csv(.x, paste0(here("data/cleaned/"), .y, ".csv")))
}

# Saves a .RDS list object with the results of the modeling process
save_results <- function(..., fname) {
  results <- list(...)
  names(results) <- as.character(substitute(list(...)))[-1]
  saveRDS(results, file = here("out/res", paste0(fname, ".RDS")))
}

# Loads lattice model for specified data set with `mod`, i.e. `mod = gmc`
load_lat_model <- function(mod) {
  mod <- deparse(substitute(mod))
  path <- here("results/lat_models", paste0(mod, "_model.tf"))
  
  res <- tf$keras$models$load_model(path)
}

# Evaluate the loaded model
eval_lat_mod <- function(mod, df) {
  
  ## train, test df
  df_name <- deparse(substitute(df))
  df_split <- rsample::initial_split(df, 
                                     prop = 0.8, 
                                     strata = "default")
  train_data <- rsample::training(df_split)
  test_data <- rsample::testing(df_split)
  
  ## extract fun
  extract_features <- function(df, label_name = "default"){
    ls <- list()
    ls[[1]] <- map(select(df, -default), `[`) %>% 
      set_names(., NULL) %>% 
      map(., ~np_array(., dtype = "float64"))
    ls[[2]] <- np_array(df[["default"]], dtype = "float64") 
    names(ls) <- c("xs", "ys")
    return(ls)
  }
  
  ## extract train test
  train_xy <- extract_features(train_data)
  test_xy <- extract_features(test_data)
  
  ## eval
  print(paste0('Lattice Test Set Evaluation for ', df_name))
  eval <- mod$evaluate(test_xy$xs, test_xy$ys)
} 

# Latex tables for the paper

# read_rds(here("results/exp_results", "results.RDS"))$res %>% 
#   select(data_set, tr_auc, ts_auc, ttr_auc, tts_auc) %>%
#   mutate(` ` = ts_auc - tts_auc) %>%
#   kable("latex", caption = "Results",
#         booktabs = T, digits = 4) %>%
#   kable_styling(latex_options = c("hold_position"), full_width = F, 
#                 position = "center") %>%
#   add_header_above(c(" " = 1, "Lattice" = 2, "Tree" = 2, "Diff" = 1), bold = T)
# 
# overview() %>%
#   kable("latex", caption = "Overview",
#         booktabs = T, digits = 2) %>%
#   kable_styling(latex_options = c("hold_position"), full_width = F, 
#                 position = "center")


# ***************************************************************************************************************************** #