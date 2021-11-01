#### Master's thesis:  
## Building an Activity Prediction Agent as part of a Multi-Agent Recommendation System for a More Efficient Energy Use in Private Households

Author: Laura Löschmann

Reviewers: Stefan Lessmann, Benjamin Fabian

Supervisor: Alona Zharova

#### Abstract

The energy sector is responsible for 60% of the global manmade greenhouse gas emissions and thus, a main factor of climate change. The energy consumption of private households amounts to approximately 30% of the total global energy consumption, causing a large share of the CO2 emissions through energy production. An intelligent demand response via load shifting increases the energy efficiency of private households by supporting residents in changing their energy consumption behaviour. This paper introduces an Activity Prediction Agent as part of a utility-based and context-aware multi-agent recommendation system that generates an activity shifting schedule for a 24-hour time horizon to either focus on CO2 emissions or energy costs savings. Using activities as the term residents describe their domestic life with, supports the users to integrate the recommendations into their daily life easier to implement the system over a long period. As a utility-based recommendation system it models user preferences depending on user availability, device usage habits and the household’s individual activities. As a context-aware system, it requires external hourly CO2 emissions and hourly price signals for the following day as well as the household’s energy consuming activities, the devices used to carry out the activities and energy consumption data on device level. The multi-agent architecture provides flexibility for adjustments and further improvements. The empirical analysis indicates that depending on the chosen recommendation focus the system can provide CO2 emissions savings of 12% and energy costs savings of 20% for the studied households. It supports stabilising the grid by flattening peak-loads and thus, helps to unburden the environment.

*Keywords*   Activity Prediction - Recommendation System - Load Shifting - Energy Efficiency

---

This repository contains the code that was developed as part of the master's thesis
as well as the activity-device mapping files necessary for the Activity Prediction Agent.
Data of the households 1 to 5 from the REFIT dataset were used (https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned).

The first four points of the notebook's agenda need to be executed first, the following points can be executed in any order. However, for the execution of the overall recommendation system the calculation of the AUC scores within the Availability Agent and the Usage Agent needs to be disabled (see comments in the corresponding NN_aval_use_agent.py file).

