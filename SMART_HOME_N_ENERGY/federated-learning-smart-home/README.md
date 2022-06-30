# An Application of Federated Learning in a Smart Home Environment
# Master Thesis

## Author: Daniel Bustillo  
## 1st Examiner: Prof. Dr. Stefan Lessmann
## 2nd Examiner: PD Dr. Benjamin Fabian
## Supervisor: Dr. Alona Zharova
## Date: 13.06.2022

### Setup
Clone this repository

Create an virtual environment and activate it

```
python -m venv thesis-env
source thesis-env/bin/activate
```

Install requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```
This repository consists of three different notebooks, each one for different data sets.

### Some results

![plotapt10](https://user-images.githubusercontent.com/52081079/176722802-f33cc3b2-5483-42e3-8168-f5293b3bc2d7.png)

### Summary
Load forecasting is essential for regulating supply and demand for energy in Smart Grids. Each participating household in the grid would greatly profit from the electricity consumption patterns from other homes in order to improve the accuracy of the forecasts. However, due to data privacy and security, it is not advisable to centralize this sensible energy consumption data from all households in a single server or entity. A novel approach called Federated Learning is a machine learning technique that trains data in a decentralized manner, without sending local data to a global server or to other clients. Using this framework, we design a system architecture that focuses on load forecasting for heterogeneous clients. Despite the fact that the results show that a centralized model still performs better than federated models, the positive take away of this paper is that a federated approach outperforms single client models, whilst preserving the privacy of their data.

Keywords: Federated Learning, Load Forecasting, Time Series Forecasting, Decentralized Learning

### Structure
```
├── README.md
├── requirements.txt                                -- required libraries
├── Data sets                                       -- stores csv file 
├── Plots                                           -- stores image files
├── apartment.ipynb                                 -- notebook for the apartments data set
├── refit.ipynb                                     -- notebook for the REFIT data set
├── smarthome.ipynb                                 -- notebook for the smarthome data set
└── preprocessing_smart.py                          -- preprocessing of the smart data set              
```
