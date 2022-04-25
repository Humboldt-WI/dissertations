# Deep Learning for Appliance Level Load Prediction

#### Master's Thesis
#### Author: Antonia Scherz
#### 1st Examiner: Prof. Dr. Stefan Lessmann
#### 2nd Examiner: PD Dr. Benjamin Fabian 
#### Supervisor: Dr. Alona Zharova
#### Date: 15.04.2022

---

This repository contains the code used for the Thesis on Deep Learning for Appliance Level Load Prediction
.

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
4. To run the notebooks datasets must be downloade from the source pages. The Notebook `Prepare_Source_Data.ipynb` containes detailed instructions on how to get the data. Alternatively the raw data is available on a gdrive. To acces the data please ask me via Mail for an invitation link, as some data should not be shared publicly.

### Usage 
- First, the notebook `prepare_source_data.ipynb` inside src folder must be executed to prepare the data for all steps later. Downloaded data should be stored in a folder called data.
- Second in order to get acces to the feature generation with VEST, this repository https://github.com/vcerqueira/vest-python needs to be downloaded and stored in a folder called vest. Then the working path should be set to this folder, in order to ensure that imports are available. (in a notebook: %cd /path/to/vest/folder/)
- Third, after the `prepare_source_data.ipynb` is executed, `data_preparation.ipynb` and `model_training.ipynb` can be run with configurations specified in the parameter setting cell in each notebook.
- Optional Notebooks can be run directly in colab via the gdrive link

### Project structure
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
