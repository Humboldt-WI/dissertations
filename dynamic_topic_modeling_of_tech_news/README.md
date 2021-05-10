### Dynamic Embedded Topic Modeling
#### An Exploration of 20 Years in Technology News


#### The following notebooks and Python files accompany the thesis and should be run in the following order:
- 01_Data-Collection.ipynb: Requests the data through The Guardian's API and saves it as .json and .csv
- 02_Exploration_Prep.ipynb: Exploration of the data, first pre-processing steps & creation of the text variable
- 03_Text-Prep.ipynb: Pre-processing of the texts
- 04_Final-Prep.ipynb: Final pre-processing steps as required by the D-ETM & pre-processing for visualisation purposes
- 05_Embeddings.ipynb: Creates (smaller) embedding matrices based on Word2Vec, GloVe and fastText
- 06_Training-Terminal.ipynb: Used as a terminal for model training. Allows explorations of progress during and after training as well as simple modifications using argparse.
  - to run main.py (Training and evaluation of different D-ETMs)
    - main.py uses:
      - detm.py (the D-ETM as defined by Dieng, Ruiz and Blei (2019))
      - utils.py (helper functions to obtain data, visualise results, find neighbours in the embedding space, label topics, obtain/save results, evaluate performance)
- 07_Visualisation.ipynb: Visualisation of results

#### Please note:
- The data must not be shared, but it can be obtained from https://open-platform.theguardian.com/.
- Directories in the code will have to be adapted.
- Most of the ouputs of the notebooks have been cleared to respect The Guardian's terms of use
- Parts of the code are adapted from publicly available sources. The notebooks and Python files contain references.