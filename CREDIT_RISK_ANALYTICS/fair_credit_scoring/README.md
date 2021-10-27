# Code Manual: Fairness in Credit Scoring Models

##### Master Thesis from Johannes Jacob




#### 1. AIF360
Copy folder aif360 into local python package folder

#### 2. Py Code
A. Change the path to file location in each script, e.g. variables output_path and syspath

B. Run following scripts in that order:

1. 01_preProcessors.py
2. 02_inProcessors.py
3. 021_adversarial_debiasing.py
4. 03_postProcessors.py
    
    

#### 3. R Code
A. Change the path to file location in each script 

B. Run following scripts in that order:

1. 03_preProcessors.R *<- Change input according to the pre-processor that should be examined.*
2. 04_inProcessors.R *<- Change input according to the in-processor that should be examined.*
3. 050_postProcessor_PyInput.R
4. 051_postProcessor_EOP.R
5. 052_postProcessor_Platt.R
6. 053_postProcessor_PyOutput.R *<- Change input according to the in-processor that should be examined.*
7. 01_Base_scenario.R
8. 02_MaxProf_scenario.R
