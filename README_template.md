# Title

**Type:** Master's Thesis / Bachelor's Thesis

**Author:** xxx

**Supervisor:** xxx (only if different from 1st or 2nd Examiner)

**1st Examiner:** xxx 

**2nd Examiner:** xxx 

**Date of submission:** 01.01.2022

## Abstract

(Short summary of motivation, contributions and results)

**Keywords**: xxx (please name at least 5 keywords / phrases).

## Working with the repo

### Dependencies / Prerequisite

Which Python version is required? 

## Setup

[Is there a virtual environment that should be activated?]

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Place the folders `Deployment_1` and `Deployment_2` from `SMARTENERGY.KOM/Raw_Data` into `data` folder

### Reproducing results

Describe steps how to reproduce your results.

### Project structure

(Here is an example from SMART_HOME_N_ENERGY, "Appliance Level Load Prediction" repo)

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file (only available via gdrive link)
├── plots                                           -- stores image files
└── src
    ├── prepare_source_data.ipynb                   -- preprocesses data
    ├── data_preparation.ipynb                      -- preparing datasets
    ├── model_tuning.ipynb                          -- tuning functions
    └── run_experiment.ipynb                        -- run experiments 
    └── plots                                       -- plotting functions                 
```
