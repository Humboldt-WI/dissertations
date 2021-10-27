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
  01preprocess.R [options]
  01preprocess.R (--dir|--web) [--write=<logical>]
  01preprocess.R (--dir|--web) [--oview --corplot]

Options:
-h --help                     Show this screen.
--dir                         Load data from directory
--web                         Load data from internet
--write=<bool>                Write cleaned data to dir [default: TRUE]
--oview                       Gives overview on all data sets
--corplot                     Example correlation plot
--oviewplot                   Example overview plot

" -> doc

opt <- docopt(doc)

# Load (and install) required packages
pkgs = c("docopt", "corrr", "here",
         "cowplot","scales", "tidyverse")
print(paste0("Loading the following packages: ", paste(pkgs, collapse=", ")))
suppressMessages(load_pkgs(pkgs))

set.seed(4321)

#print(opt) 

# Preprocess ****************************************************************************************************************** #

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

# Variable selection ---------------------------------------------------------------------------------------------------------- #

# Correlation plot for GER
ger_cor <- function(){  
  pdf(here("out/plots","ov_plot_ger.pdf"),
  height = 10,
  #width = 10,
  paper = "a4")
  
  data_raw$ger_raw %>%
    dplyr::mutate(
      across(where(is.character), as.factor)) %>%
    dplyr::rename(default = credit_risk) %>%
    corplot()
  
  dev.off()
}

# Call it all ***************************************************************************************************************** #


  # Load data
  data_from <- ifelse(opt$dir, TRUE, FALSE)
  data_raw <- suppressWarnings(load_raw_data(dir = data_from))
  
  # Clean data
  data_clean <- cleaning_data(data_raw, write = as.logical(opt$write))
  if (opt$write) cat("\n", "\n", "Cleanded data saved in /data/cleaned")
  # EDA
  ## Overview
  if (opt$oview) overview()
  ## Corplot for GER
  if (opt$corplot) {
    ger_cor() 
    cat("\n", "\n", "Corplot saved in /out/plots")
  }
  ## Overview plot on GER
  if (opt$oviewplot) {
    ploting_df(data_clean$ger, view = FALSE, save = TRUE) 
    cat("\n", "\n", "Overview plot saved in /out/plots")
  }

  cat("\n")

