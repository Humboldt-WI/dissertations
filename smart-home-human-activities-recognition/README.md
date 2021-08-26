# Human Activity Recognition in Smart Home environment using Machine Learning models

> **Bachelor thesis - Doan Hai Anh Bui** 
---

This repository contains the code used for Bui's bachelor thesis.

### Prerequisite

- Python 3.7 or higher

### Setup
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

### Usage 
- The script `dataset.py` inside src folder must be executed right after the setup is done to prepare the data for all steps later.
- After the `dataset.py` is executed, other scripts including notebooks can be executed out of order (for more information about jupyter notebook refer to this [link](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html))


### Project structure
```bash
├── README.md
├── requirements.txt                                -- required libraries
├── setup.py
└── src
    ├── data                                        -- stores csv file
    ├── dataset.py                                  -- preprocesses data
    ├── __init__.py
    ├── notebooks                                   -- stores notebook scripts
    │   ├── data_exploration_deployment_1.ipynb     -- deployment 1 data exploration 
    │   ├── data_exploration_deployment_2.ipynb     -- deployment 2 data exploration
    │   ├── LSTM_deployment_1.ipynb                 -- training LSTM model for deployment 1
    │   ├── LSTM_deployment_2.ipynb                 -- training LSTM model for deployment 2
    │   ├── modeling_deployment_1.ipynb             -- training action recognition models for deployment 1
    │   ├── modeling_deployment_2.ipynb             -- training action recognition models for deployment 2
    │   ├── overview_deployment_1.ipynb             -- overview of all deployment 1 datasets
    │   ├── overview_deployment_2.ipynb             -- overview of all deployment 2 datasets
    │   └── results                                 -- stores training results
    ├── ratio_split_experiment.py                   -- uses to determine the best data split ratio
    └── utils                                       -- stores helpful functions
        ├── __init__.py
        └── processing.py                       
```
